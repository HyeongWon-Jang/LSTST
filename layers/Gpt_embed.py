import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, time):
        time = time[time != -100]
        time = time - 1
        return self.pe[:, time]
    



class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_width, padding=0, bias=True):
        super(Conv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.kernel_width = kernel_width
        self.padding=padding
        self.kernel = nn.Linear(kernel_width * in_channels, out_channels, bias=bias)

        with torch.no_grad():
            self.kernel.weight.copy_(torch.randn(out_channels, kernel_width * in_channels))

    def forward(self, x):
        x = F.pad(x, pad=(self.padding,self.padding))
        l = [self.kernel(x[:, :, i - self.kernel_width: i].reshape(x.shape[0], self.in_channels * self.kernel_width)) for i in range(self.kernel_width, x.shape[2]+1, self.stride)]
        return torch.stack(l, dim=1)


class TokenEmbedding(nn.Module):
    def __init__(self, feat_dim, d_model, kernel_width=3,stride=1,padding=1):
        super(TokenEmbedding, self).__init__()
 

        #self.tokenConv = Conv1d(in_channels=feat_dim, out_channels=d_model, stride=stride, kernel_width=kernel_width, padding=padding, bias=False)
        self.tokenConv = torch.nn.Conv1d(in_channels=feat_dim, out_channels=d_model, kernel_size=kernel_width, stride=stride, padding=padding, padding_mode='circular', bias=False)
    
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')
                
    def forward(self, x):  
        # x = x / 100.0
        # my conv1d
        # x = self.tokenConv(x.permute(0,2,1))
        x = self.tokenConv(x.permute(0,2,1)).permute(0,2,1)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, feat_dim, d_ff, d_model, max_len, embed_type='fixed', freq='h', dropout=0.1, kernel_width=3,stride=1,padding=1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(feat_dim=feat_dim, d_model=d_ff, kernel_width=kernel_width,stride=stride,padding=padding)
        self.position_embedding = PositionalEmbedding(d_model=d_ff, max_len=max_len)
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, time):
        #x = self.normalization(x, 'norm')
        B, C, L = x.shape 
        x = self.value_embedding(x) + self.position_embedding(x, time)
        result = torch.zeros((x.shape[0], x.shape[1], self.d_model)).to(x.device)
        result[:, :, :self.d_ff] = x
        return self.dropout(result)
