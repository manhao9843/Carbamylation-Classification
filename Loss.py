from torch import nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, device, gamma=2.0, alpha_1=0.25):
        super(FocalLoss, self).__init__()
        self.device = device
        self.gamma = gamma
        self.alpha = torch.tensor([1-alpha_1, alpha_1])
    
    def forward(self, inputs, targets):
        # get batch_size
        #batch_size = inputs.shape[0]
        
        # one-hot encoding the labels
        idx = targets.view(-1,1)
        one_hot_key = torch.nn.functional.one_hot(idx, 2).squeeze(1).to(self.device)
        
        # softmax
        probs = nn.Softmax(dim=1)(inputs.float()).to(self.device)
        
        # compute loss
        loss = -one_hot_key * torch.pow((1-probs), 2) * probs.log() * (self.alpha.to(self.device))
        loss = loss.sum()
        
        return loss

