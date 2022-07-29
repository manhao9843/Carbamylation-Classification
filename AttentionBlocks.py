import torch
from torch import nn

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)

        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # queries: [batch_size, num_queries, h]
        # keys: [batch_size, num_keys, h]
        queries, keys = self.W_q(queries), self.W_k(keys)
    
        # queries: [batch_size, num_queries, 1.      , h]
        # keys:    [batch_size, 1.         , num_keys, h]
        # features:[batch_size, num_queries, num_keys, h]
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # scores: [batch_size, num_queries, num_keys]
        # values: [batch_size, num_keys,    len_of_value]
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = nn.functional.softmax(scores,dim=-1)
        # [batch_size, num_queries, len_of_value]
        return torch.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)

        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # queries: [batch_size, num_queries, h]
        # keys: [batch_size, num_keys, h]
        queries, keys = self.W_q(queries), self.W_k(keys)

        # scores: [batch_size, num_queries, num_keys]
        scores = torch.bmm(queries, torch.transpose(keys, 1, 2))
        self.attention_weights = nn.functional.softmax(scores,dim=-1)

        # [batch_size, num_queries, len_of_value]
        return torch.bmm(self.dropout(self.attention_weights), values)

