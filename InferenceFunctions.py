from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import torch.nn
from torch import nn
import torch
import plotly.graph_objects as go
import numpy as np

def inference(model, dataloader, device, with_returns=False):
    
    model.eval()
    pred_list = list()
    true_list = list()

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            logits = model(X,0)
            prob = nn.Softmax(dim=1)(logits)
            y_pred = prob.argmax(1)
            pred_list += y_pred.tolist()
            true_list += y.tolist()
    
    print(confusion_matrix(true_list,pred_list))
    print(classification_report(true_list,pred_list))
    print(accuracy_score(true_list, pred_list))
    
    if with_returns:
        return pred_list, true_list


def plot_loss(train_loss_storage, test_loss_storage):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(len(train_loss_storage))), y=train_loss_storage,
                    mode='lines',
                    name='Training Loss'))
    
    fig.add_trace(go.Scatter(x=np.array(range(len(train_loss_storage))), y=test_loss_storage,
                    mode='lines',
                    name='Validation Loss'))
    
    fig.update_layout(xaxis_title="Epochs",yaxis_title="Loss")
    fig.show()
    

def plot_acc(train_acc_storage, test_acc_storage):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(range(len(train_acc_storage))), y=train_acc_storage,
                    mode='lines',
                    name='Training Loss'))
    
    fig.add_trace(go.Scatter(x=np.array(range(len(train_acc_storage))), y=test_acc_storage,
                    mode='lines',
                    name='Validation Loss'))
    
    fig.update_layout(xaxis_title="Epochs",yaxis_title="Acc")
    fig.show()