from ResidualBlocks import ResidualFeedForward
from AttentionBlocks import AdditiveAttention
import torch
from torch import nn

class AttentionBasedNN(nn.Module):
    def __init__(self, vec_size, n_hidden, dropout):
        super(AttentionBasedNN, self).__init__()
        self.attention1 = AdditiveAttention(vec_size,vec_size,n_hidden,dropout)
        self.layernorm1 = torch.nn.LayerNorm(vec_size, elementwise_affine = False)
        self.residual1 = ResidualFeedForward(vec_size, n_hidden)
        
        self.attention2 = AdditiveAttention(vec_size,vec_size,n_hidden,dropout)
        self.layernorm2 = torch.nn.LayerNorm(vec_size, elementwise_affine = False)
        self.residual2 = ResidualFeedForward(vec_size, n_hidden)
        
        self.attention3 = AdditiveAttention(vec_size,vec_size,n_hidden,dropout)
        self.layernorm3 = torch.nn.LayerNorm(vec_size, elementwise_affine = False)
        self.residual3 = ResidualFeedForward(vec_size, n_hidden)
        
        self.globalattention = AdditiveAttention(vec_size,vec_size,n_hidden,dropout)
        self.layernorm4 = torch.nn.LayerNorm(vec_size, elementwise_affine = False)
    
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(vec_size, 32),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(32,12),
                nn.ReLU(),
                nn.Dropout(p=0.2),
                nn.Linear(12,2),)
    
    def forward(self, X, valid_lens, lys_pos):
        """
        valid_lens: a list of integers that indicate how many amino acid vectors should be counted for each sample
        
        lys_pos: an integer indicating the position of lys at dimension 1 of the input tensor with shape [batch_size, max_num_aminos, vec_size]
        
        """
        X = X.to(torch.float32)
        
        # 1. three times attention + residual feedforward
        X = self.attention1(X,X,X,valid_lens) + X
        X = self.layernorm1(X)
        X = self.residual1(X)
        
        X = self.attention2(X,X,X,valid_lens) + X
        X = self.layernorm2(X)
        X = self.residual2(X)
        
        X = self.attention3(X,X,X,valid_lens) + X
        X = self.layernorm3(X)
        X = self.residual3(X)
        
        # 2. global attention for lys
        X_lys = torch.select(X, 1, lys_pos).unsqueeze(1)
        X_lys = self.globalattention(X_lys,X,X,valid_lens) + X_lys
        X_lys = self.layernorm4(X_lys)
        
        logits = self.linear_relu_stack(X_lys).squeeze(1)
        return logits 


class MLP(nn.Module):
    def __init__(self, vec_size, dropout):
        super(MLP, self).__init__()
        self.linear_relu_stack = nn.Sequential(
                nn.Linear(vec_size, 48),
                nn.ReLU(),
                nn.Linear(48,32),
                nn.ReLU(),
                nn.Linear(32,12),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(12,2),)
    def forward(self, X):
        X = X.to(torch.float32)
        logits = self.linear_relu_stack(X)
        return logits 