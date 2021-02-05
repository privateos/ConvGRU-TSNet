import torch
import torch.nn as nn
import torch.nn.functional as F

class Pad(nn.Module):
    def __init__(self, pad, mode='constant', value=0):
        super(Pad, self).__init__()
        self.pad = pad
        self.mode = mode
        self.value = value
    
    def forward(self, x):
        return F.pad(x, self.pad, self.mode, self.value)
class ConvGRU1DCell(nn.Module):
    #input.shape (batch_size, in_channels, in_features)
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, groups=1):
        super(ConvGRU1DCell, self).__init__()
        #padding_left = kernel_size//2
        #padding_right = kernel_size - 1 - padding_left
        self.z_wx = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, groups=groups)
        self.z_uh = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, dilation=dilation, groups=groups)
        self.r_wx = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, groups=groups)
        self.r_uh = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, dilation=dilation, groups=groups)
        self.h_wx = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, groups=groups)
        self.h_uh = nn.Conv1d(out_channels, out_channels, kernel_size, stride=1, dilation=dilation, groups=groups)
        self.h_padding = Pad(((kernel_size - 1)*dilation, 0))
 
    def forward(self, x_t, h_t_1=None):
        h_t = None
        if h_t_1 is None:
            z_t = torch.sigmoid(self.z_wx(x_t))
            #r_t = torch.sigmoid(self.r_wx(x_t))
            h__t = torch.tanh(self.h_wx(x_t))
            h_t = (1 - z_t)*h__t
        else:
            padding_h_t_1 = self.h_padding(h_t_1)
            z_t = torch.sigmoid(self.z_wx(x_t) + self.z_uh(padding_h_t_1))
            r_t = torch.sigmoid(self.r_wx(x_t) + self.r_uh(padding_h_t_1))
            h__t = torch.tanh(self.h_wx(x_t) + r_t*self.h_uh(padding_h_t_1))
            #print(h__t.shape, z_t.shape, h_t_1.shape)

            h_t = (1 - z_t)*h__t + z_t*h_t_1

        return h_t
class ConvGRU1D(nn.Module):
    # input.shape = (batch_size, seq_len, in_channels, in_features)
    # output.shape = (batch_size, seq_len, out_channels, in_features)
    def __init__(self,in_channels, out_channels, kernel_size, dilation=1, groups=1, dropout=0.0):
        super(ConvGRU1D, self).__init__()
        self.x_padding = Pad(((kernel_size - 1)*dilation, 0))
        self.ConvGRU1DCell = ConvGRU1DCell(in_channels, out_channels, kernel_size, dilation=dilation, groups=groups)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, h_0=None):
        x = self.x_padding(x)
        seq_len = x.size(1)

        h_t_1 = h_0

        h_list = []

        for i in range(seq_len):
            x_t = x[:, i, :, :]
            h_t = self.ConvGRU1DCell(x_t, h_t_1)
            
            h_list.append(torch.unsqueeze(h_t, 1))
            h_t_1 = h_t
        
        h = torch.cat(h_list, 1)
        h = self.dropout(h)
        return h
class GRU(nn.Module):
    #x.shape = (batch_size, time, input_size)
    #y.shape = (batch_size, time, hidden_size)
    def __init__(self, input_size, hidden_size, dropout=0.0):
        super(GRU, self).__init__()
        self.nn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        y, h_n = self.nn(x)
        return self.dropout(y)
class Reshape(nn.Module):
    def __init__(self, new_shape):
        super(Reshape, self).__init__()
        self.new_shape = new_shape
    
    def forward(self, x):
        return torch.reshape(x, self.new_shape)
class GetLastTime(nn.Module):
    #(batch_size, time, feature)
    def __init__(self):
        super(GetLastTime, self).__init__()
    
    def forward(self, x):
        return x[:, -1, :]
class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim
    
    def forward(self, x):
        return torch.squeeze(x, self.dim)
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    
    def forward(self, x):
        return torch.transpose(x, self.dim0, self.dim1)
class AR(nn.Module):
    #input.shape = (batch_size, seq_len, input_size)
    #output.shape = (batch_size, input_size)
    def __init__(self, seq_len):
        super(AR, self).__init__()
        self.nn = nn.Sequential(
            #(batch_size, seq_len, input_size)
            Transpose(1, 2),
            #(batch_size, input_size, seq_len)
            nn.Linear(seq_len, 1),
            #(batch_size, input_size, 1)
            Squeeze(2)
            #(batch_size, input_size)
        )
    
    
    def forward(self, x):#x.shape = (batch_size, seq_len, input_size)
        return self.nn(x)

class ConvMLPAttention(nn.Module):
    #x.shape = (batch_size, x_rows, x_columns):
    #q.shape = (batch_size, q_columns)
    def __init__(self, x_columns, q_columns):
        super(ConvMLPAttention, self).__init__()
        self.conv = nn.Sequential(
            #(batch_size, x_rows, x_columns)
            Transpose(1, 2),
            #(batch_size, x_columns, x_rows)
            nn.Conv1d(x_columns, x_columns, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
            #(batch_size, x_columns, x_rows)
            Transpose(1, 2),
            #(batch_size, x_rows, x_columns)
        )

        self.mlp = nn.Sequential(
            #(batch_size, x_rows, x_columns + q_columns)
            nn.Linear(x_columns + q_columns, (x_columns + q_columns)//2),
            nn.Tanh(),
            #(batch_size, x_rows, (x_columns + q_columns)//2)
            nn.Linear((x_columns + q_columns)//2, 1),
            nn.Tanh()
            #(batch_size, x_rows, 1)
        )
    
    #x.shape = (batch_size, x_rows, x_columns)
    #q.shape = (batch_size, q_columns)
    def forward(self, x, q):
        x_rows = x.size(1)
        #conv_x.shape = (batch_size, x_rows, x_columns)
        conv_x = self.conv(x)

        #q1.shape = (batch_size, 1, q_columns)
        q1 = torch.unsqueeze(q, 1)

        #q2.shape = (batch_size, x_rows, q_columns)
        q2 = q1.repeat(1, x_rows, 1)

        #conv_x_q.shape = (batch_size, x_rows, x_columns + q_columns)
        conv_x_q = torch.cat([conv_x, q2], 2)

        #attention.shape = (batch_size, x_rows, 1)
        attention = self.mlp(conv_x_q)
        return attention

class FeatureExtraction(nn.Module):
    def __init__(self, n_time, n_feature):
        super(FeatureExtraction, self).__init__()
        #tf_x.shape = (batch_size, n_time - 1, n_feature)
        #tf_q.shape = (batch_size, n_feature)
        self.tf = ConvMLPAttention(n_feature, n_feature)
        #sf_x.shape = (batch_size, n_feature, n_time - 1)
        #sf_q.shape = (batch_size, n_feature)
        self.sf = ConvMLPAttention(n_time - 1, n_feature)

    #x.shape = (batch_size, n_time, n_feature)
    def forward(self, x):
        tf_x = x[:, :-1, :]
        tf_q = sf_q = x[:, -1, :]
        sf_x = torch.transpose(tf_x, 1, 2)

        #sf_attention_score.shape = (batch_size, n_time - 1, 1)
        tf_attention_score = self.tf(tf_x, tf_q)
        #sf_attention_score.shape = (batch_size, n_feature, 1)
        sf_attention_score = self.sf(sf_x, sf_q)

        #tf_attention.shape = (batch_size, n_feature)
        tf_attention = torch.mean(tf_x*tf_attention_score, 1)
        #sf_attentin.shape = (batch_size, n_time - 1)
        sf_attention = torch.mean(sf_x*sf_attention_score, 1)

        #attention.shape = (batch_size, n_feature + n_time - 1 + n_feature)
        #                = (batch_size, n_feature*2 + n_time - 1)
        attention = torch.cat([tf_attention, sf_attention, tf_q], 1)
        return attention

class ConvGRUTSNet(nn.Module):
    #input.shape = (batch_size, seq_len, in_features) = (batch_size, period_len*period_times, in_features)
    #output.shape = (batch_size, in_features)
    def __init__(self, seq_len, in_features, period_len, 
    conv_kernel_size, conv_rnn_hidden, rnn_hidden, out_features):
        super(ConvGRUTSNet, self).__init__()
        period_times = seq_len//period_len
        out_channels = conv_rnn_hidden
        #conv_out_features = period_len - conv_kernel_size + 1

        self.main_branch = nn.Sequential(
            #(batch_size, seq_len, n_feature)
            Reshape((-1, period_times, period_len, in_features)),
            #(batch_size, period_times, period_len, in_features)
            Transpose(2, 3),
            #(batch_size, period_times, in_features, period_len)
            ConvGRU1D(in_features, out_channels, conv_kernel_size),
            #(batch_size, period_times, out_channels, period_len)
            Transpose(2, 3),
            #(batch_size, period_times, period_len, out_channels)
            Reshape((-1, period_times, period_len*out_channels)),
            #(batch_size, period_times, period_len*out_channels)
            GetLastTime(),
            #(batch_size, period_len*out_channels)
            Reshape((-1, period_len, out_channels)),
            #(batch_size, period_len, out_channels)
            GRU(out_channels, rnn_hidden),
            #(batch_size, period_len, rnn_hidden)
            FeatureExtraction(period_len, rnn_hidden),
            #(batch_size, rnn_hidden*2 + period_len - 1)
            nn.Linear(rnn_hidden*2 + period_len - 1, out_features)
            #(batch_size, out_features)
        )
                            
        self.linear_branch = nn.Sequential(
            #(batch_size, seq_len, in_features)
            Reshape((-1, period_times, period_len*in_features)),
            #(batch_size, period_times, period_len*in_features)
            GetLastTime(),
            #(batch_size, period_len*in_features)
            Reshape((-1, period_len, in_features)),
            #(batch_size, period_len, in_features)
            AR(period_len),
            #(batch_size, in_features)
            nn.Linear(in_features, out_features) if in_features != out_features else nn.Sequential()
            #(batch_size, out_features)
        )
                            
    def forward(self, x):
        main_branch = self.main_branch(x)
        linear_branch = self.linear_branch(x)
        result = main_branch + linear_branch
        return result

if __name__ == '__main__':
    test = ConvGRUTSNet(10, 13, 5, 3, 11, 30, out_features=12)
    x = torch.randn((3, 10, 13))
    y = test(x)
    print(y.shape)