import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class MyDataset(Dataset):
    def __init__(self, encoded_list, labels):
        self.features = encoded_list
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self,idx):
        data = self.features[idx]
        label = self.labels[idx]
        
        return data, label

def collate_batch(batch):
    label_list, text_list, = [], []
    # essential for masked softmax
    valid_lens = []
   
    for (_text,_label) in batch:
        label_list.append(_label)
        text_list.append(_text)
        valid_lens.append(_text.shape[0])
   
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, batch_first=True, padding_value=0)
   
    return text_list,label_list, torch.tensor(valid_lens)

