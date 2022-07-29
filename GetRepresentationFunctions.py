import torch
import torch.nn
import numpy as np

def GetRepresentation(dataloader, model, device):
    embeddings = list()
    for X, y in dataloader:
        X = X.to(torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            X = model.attention1(X,X,X) + X
            X = model.layernorm1(X)
            X = model.residual1(X)
            
            X = model.attention2(X,X,X) + X
            X = model.layernorm2(X)
            X = model.residual2(X)
            
            X = model.attention3(X,X,X) + X
            X = model.layernorm3(X)
            X = model.residual3(X)
            
            X_lys = torch.select(X, 1, 0).unsqueeze(1)
            X_lys = model.globalattention(X_lys,X,X) + X_lys
            X_lys = model.layernorm4(X_lys)
            
            embeddings += model.linear_relu_stack[:4](X_lys).squeeze(1).tolist()
    return np.array(embeddings)

