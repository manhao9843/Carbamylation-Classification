import torch

def train(dataloader, model, loss_fn, optimizer, loss_storage, acc_storage, device, scheduler=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    train_loss, correct = 0, 0
    
    for batch, (X,y,valid_lens) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X,valid_lens,0)
        correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    loss_storage.append(train_loss)
    acc_storage.append(correct)

def test(dataloader, model, loss_fn, loss_storage, acc_storage, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y, valid_lens in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X,valid_lens,0)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    loss_storage.append(test_loss)
    acc_storage.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_mlp(dataloader, model, loss_fn, optimizer, loss_storage, acc_storage, device, scheduler=None):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    model.train()
    train_loss, correct = 0, 0
    
    for batch, (X,y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        correct += (pred.argmax(1)==y).type(torch.float).sum().item()
        loss = loss_fn(pred, y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        if batch % 10 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    
    train_loss /= num_batches
    correct /= size
    loss_storage.append(train_loss)
    acc_storage.append(correct)

def test_mlp(dataloader, model, loss_fn, loss_storage, acc_storage, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    model.eval()
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1)==y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    loss_storage.append(test_loss)
    acc_storage.append(correct)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")