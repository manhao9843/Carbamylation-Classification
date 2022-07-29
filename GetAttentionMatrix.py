import torch
import torch.nn
from torch import nn
import matplotlib.pyplot as plt
from matplotlib import ticker

def GetAttentionScores(input_, model, which_attention):
    
    # check dimension: [batch_size, num_amino, 44]
    if len(input_.shape) != 3:
        X = input_.unsqueeze(0)
    else:
        X = input_
    
    model.eval()
    with torch.no_grad():
        logits = model(X,0)
        prob = nn.Softmax(dim=1)(logits)
        y_pred = prob.argmax(1).item()

        X = model.attention1(X,X,X) + X
        score1 = model.attention1.attention_weights
        if which_attention == 0:
            return score1, y_pred
        
        X = model.layernorm1(X)
        X = model.residual1(X)
        X = model.attention2(X,X,X) + X
        score2 = model.attention2.attention_weights
        if which_attention == 1:
            return score2, y_pred

        X = model.layernorm2(X)
        X = model.residual2(X)
        X = model.attention3(X,X,X) + X
        score3 = model.attention3.attention_weights
        if which_attention == 2:
            return score3, y_pred

        X = model.layernorm3(X)
        X = model.residual3(X)
        X_lys = torch.select(X, 1, 0).unsqueeze(1)
        X_lys = model.globalattention(X_lys,X,X) + X_lys
        score4 = model.globalattention.attention_weights
        if which_attention == 3:
            return score4, y_pred
        
        return 'Attention not found.'

    
def ShowAttentionScores(input_, actual_label, model, which_attention, decode_dict, size=10):
    try:
        scores, y_pred = GetAttentionScores(input_, model, which_attention)
    except:
        return 'Please select attention num 0,1,2,3.'
    
    if len(input_.shape) != 2:
        X = input_.squeeze(0)
    else:
        X = input_
    
    # get the identity of amino acids
    amino_acids = list()
    X = X[:,:20]
    for i in X:
        amino_ID = ((i == 1).nonzero(as_tuple=True)[0].item())
        amino_name = decode_dict[amino_ID]
        amino_acids.append(amino_name)
    
    # show the matrix
    fig = plt.figure(figsize = [size,size])
    ax = fig.add_subplot(111)
    cax = ax.matshow(scores.squeeze(0).detach().numpy(), cmap='Greens')
    #fig.colorbar(cax)
    
    ax.set_xticklabels([''] + amino_acids, rotation=90)
    ax.set_yticklabels([''] + amino_acids)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    # add title
    ax.set_title(f'True Label: {actual_label}, Predicted Label: {y_pred}')
    ax.set_xlabel('Keys')
    ax.set_ylabel('Queries')
    plt.show()

