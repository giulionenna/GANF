
#%%
import torch.nn as nn
import torch.nn.functional as F
from models.NF import MAF, RealNVP
import torch
from torch.nn.utils import weight_norm

class GNN(nn.Module):
    """
    The GNN module applied in GANF
    """
    def __init__(self, input_size, hidden_size):

        super(GNN, self).__init__()
        self.lin_n = nn.Linear(input_size, hidden_size)
        self.lin_r = nn.Linear(input_size, hidden_size, bias=False)
        self.lin_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, h, A):
        ## A: K X K
        ## H: N X K  X L X D

        h_n = self.lin_n(torch.einsum('nkld,kj->njld',h,A))
        h_r = self.lin_r(h[:,:,:-1])
        h_n[:,:,1:] += h_r
        h = self.lin_2(F.relu(h_n))

        return h

#### TCN Implementation from (https://github.com/locuslab/TCN/blob/master/TCN/tcn.py)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, output_size, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        if len(x.shape)==3:
            x = self.network(x.transpose(1,2)).transpose(1,2)
            x = self.linear(x)
        else:
            x = self.network(x.unsqueeze(1)).transpose(1,2)
            x = self.linear(x).squeeze(2)
        return x

class GANF(nn.Module):

    def __init__ (self,n_features,  n_blocks, input_size, hidden_size, n_hidden ,dropout = 0.1, model="MAF", batch_norm=True, use_tcn = True, use_gru = True, dim_red = True):
        super(GANF, self).__init__()
        if dim_red:
            reduction = nn.Sequential(
                nn.Linear(n_features, 50),
                nn.LeakyReLU()
            )
            n_features = 50
        
        if use_gru:
           self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,batch_first=True, dropout=dropout)
        else: 
            self.rnn = nn.LSTM(input_size=input_size,hidden_size=hidden_size,batch_first=True, dropout=dropout)
        self.gcn = GNN(input_size=hidden_size, hidden_size=hidden_size)
        if model=="MAF":
            self.nf = MAF(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm,activation='tanh')
        else:
            self.nf = RealNVP(n_blocks, input_size, hidden_size, n_hidden, cond_label_size=hidden_size, batch_norm=batch_norm)
        
        self.use_tcn = use_tcn
        if use_tcn:
            self.tcn = TemporalConvNet(n_features, n_features, [150, 150, 150])    

    def forward(self, x, A):

        return self.test(x, A).mean()

    def test(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        if self.use_tcn:
            #tcn takes as input a tensor N x L x K where N is batch size L is window length and K is features. Outputs a tensor with the same shape
            #all reshaping and transpositions are performed in order to accomodate the input formats of the other modules
            x = x.transpose(1,2).squeeze()
            x = self.tcn(x)
            x = x.unsqueeze(dim=-1).transpose(1,2)
        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=1)

        return log_prob
    
    def locate(self, x, A):
        # x: N X K X L X D 
        full_shape = x.shape

        # reshape: N*K, L, D
        x = x.reshape((x.shape[0]*x.shape[1], x.shape[2], x.shape[3]))
        h,_ = self.rnn(x)

        # resahpe: N, K, L, H
        h = h.reshape((full_shape[0], full_shape[1], h.shape[1], h.shape[2]))


        h = self.gcn(h, A)

        # reshappe N*K*L,H
        h = h.reshape((-1,h.shape[3]))
        x = x.reshape((-1,full_shape[3]))

        log_prob = self.nf.log_prob(x,h).reshape([full_shape[0],full_shape[1],-1])#*full_shape[1]*full_shape[2]
        log_prob = log_prob.mean(dim=2)

        return log_prob
