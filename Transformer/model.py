import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model : int,vocab_size : int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self,x):
        return self.embedding(x)*math.sqrt(self.d_model)
    

class PositionalEncoding(nn.Module):

    def __init__(self,d_model : int,seq_len : int, dropout : float):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len,d_model)

        position = torch.arange(0,seq_len,d_type = torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0,d_model,2).float()*(-math.log(1000.0) /d_model))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe',pe)

    def forward(self,x):
        x =  x + (self.pe[:,:x.shape[1],:]).requires_grad(False)
        return self.dropout(x)
    


class LayerNormalization(nn.Module):
    def __init__(self,eps :float = 1e-6):
        super().__init__()
        self.eps = eps

        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim = True)
        std_dev = x.std(dim = -1,keepdim = True)

        return self.alpha*(x-mean)/(std_dev + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    
    def __init__(self, d_model : int, d_ff :int, dropout : float ):
        super().__init__()

        self.linear1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff,d_model)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)

        return x

class MultiHeadAttentionBlock(nn.Module):
    
    def __init(self,d_model : int, h : int,dropout : float):
        super().__init__()

        self.d_model = d_model
        self.h = h
        self.dropout = nn.Dropout(dropout)

        assert self.d_model % h == 0, "d_model is not divisible by h"

        self.d_k = self.d_model//h
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query,key,value,mask,dropout : nn.Dropout):
        d_k = query.shape[-1]
        attention_scores = (query @ key.transpose(-2,1)/math.sqrt(d_k))
        
        if mask is not None:
            attention_scores.masked_fill(mask == 0,-1e9)
        attention_scores  = attention_scores.softmax(dim = -1)
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores
        
    def forward(self,q,k,v,mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        query = query.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        value = value.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)
        key = key.view(query.shape[0],query.shape[1],self.h,self.d_k).transpose(1,2)

        attention_scores = MultiHeadAttentionBlock.attention(query,key,value,mask,self.dropout)

        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)
    

class ResidualConnection(nn.Module):

    def __init__(self,dropout : float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self,x,sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    