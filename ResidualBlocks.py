from torch import nn

class ResidualFC(nn.Module):
    def __init__(self, input_length, num_hidden, **kwargs):
        super(ResidualFC, self).__init__(**kwargs)
        self.linear1 = nn.Linear(input_length, num_hidden, bias=False)
        self.linear2 = nn.Linear(num_hidden, input_length, bias=False)
        self.activation1 = nn.ReLU()
        self.activation2 = nn.ReLU()
        #self.layernorm = torch.nn.LayerNorm(input_length, elementwise_affine = False)

    def forward(self, X):
        residual = X
        X = self.linear1(X)
        X = self.activation1(X)
        X = self.linear2(X) + residual
        X = self.activation2(X)
        #X = self.layernorm(X)
        return X

