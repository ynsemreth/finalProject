import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):

    def __init__(self, input_channels, hidden_dim):

        super(ConvLSTMCell, self).__init__()        
        self.kernel_size = 3
        self.padding = 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels + hidden_dim, out_channels=4 * hidden_dim, kernel_size=self.kernel_size, padding=self.padding), 
            nn.GroupNorm(4 * hidden_dim, 4 * hidden_dim))

    def forward(self, x, hidden):
        
        h, c = hidden
        conv_output = self.conv(torch.cat([x, h], dim=1))
        i, f, g, o = torch.chunk(conv_output, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = torch.mul(f, c) + torch.mul(i, g)
        h_next = torch.mul(o, torch.tanh(c_next))

        return h_next, (c_next, h_next)


class ConvLSTM_Model(nn.Module):

    def __init__(self, args):

        super(ConvLSTM_Model, self).__init__()
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cells, self.bns = [], []
        self.batch_size = args.batch_size // args.gpu_num
        self.img_size = (args.img_size, args.img_size)
        self.n_layers = args.num_layers
        self.frame_num = args.frame_num
        self.input_dim = args.input_dim
        self.hidden_dim = args.hidden_dim
        
        self.linear_conv = nn.Conv2d(in_channels=self.hidden_dim, out_channels=self.input_dim, kernel_size=1, stride=1, padding=0)
        
        for i in range(self.n_layers):
            input_dim = self.input_dim if i == 0 else self.hidden_dim
            hidden_dim = self.hidden_dim
            self.cells.append(ConvLSTMCell(input_dim, hidden_dim))
            self.bns.append(nn.BatchNorm2d(num_features=self.hidden_dim))

        self.cells = nn.ModuleList(self.cells)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, X, hidden=None):
        
        if hidden is None:
            hidden = self.init_hidden(batch_size=self.batch_size, img_size=self.img_size)
        
        predict = []
        inputs_x = None
        
        # Hidden state updates
        for t in range(X.size(1)):
            inputs_x = X[:, t, :, :, :].to(self.device)
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)

        inputs_x = X[:, -1, :, :, :].to(self.device)
        for t in range(X.size(1)):
            for i, layer in enumerate(self.cells):
                inputs_x, hidden[i] = layer(inputs_x, hidden[i])
                inputs_x = self.bns[i](inputs_x)
                
            inputs_x = self.linear_conv(inputs_x)
            predict.append(inputs_x)
        
        predict = torch.stack(predict, dim=1)   

        return torch.sigmoid(predict)
    
    def init_hidden(self, batch_size, img_size):
        h, w = img_size
        hidden_state = (torch.zeros(batch_size, self.hidden_dim, h, w, device=self.device),
                        torch.zeros(batch_size, self.hidden_dim, h, w, device=self.device))
        states = [] 
        for i in range(self.n_layers):
            states.append(hidden_state)
        return states
