import torch
from torch import nn

# codes from github.com/d2l-ai/d2l-zh
def sequence_mask(X, valid_len, value=0):
    maxlen = X.size(1)
    mask = torch.arange((maxlen), dtype=torch.float32)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X

def masked_softmax(X, valid_lens):
    if valid_lens is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.reshape(-1)
        X = sequence_mask(
                X.reshape(-1, shape[-1]), valid_lens, value = -1e6)
    return nn.functional.softmax(X.reshape(shape), dim=-1)

class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)

        self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.W_v = nn.Linear(num_hiddens, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        # queries: [batch_size, num_queries, n_hiddens]
        # keys: [batch_size, num_keys, n_hiddens]
        queries, keys = self.W_q(queries), self.W_k(keys)
    
        # queries: [batch_size, num_queries, 1.      , n_hiddens]
        # keys:    [batch_size, 1.         , num_keys, n_hiddens]
        # features:[batch_size, num_queries, num_keys, n_hiddens]
        features = queries.unsqueeze(2) + keys.unsqueeze(1)
        features = torch.tanh(features)

        # scores: [batch_size, num_queries, num_keys]
        scores = self.W_v(features).squeeze(-1)
        self.attention_weights = masked_softmax(scores, valid_lens)
        
        # values: [batch_size, num_keys,    len_of_value]
        # output: [batch_size, num_queries, len_of_value]
        return torch.bmm(self.dropout(self.attention_weights), values)