import torch
import torch.nn
import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objects as go

def GetRepresentation(dataloader, model, device):
    embeddings = list()
    for X, y, valid_lens in dataloader:
        X = X.to(torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            X = model.attention1(X,X,X,valid_lens) + X
            X = model.layernorm1(X)
            X = model.residual1(X)
            
            X = model.attention2(X,X,X,valid_lens) + X
            X = model.layernorm2(X)
            X = model.residual2(X)
            
            X = model.attention3(X,X,X,valid_lens) + X
            X = model.layernorm3(X)
            X = model.residual3(X)
            
            X_lys = torch.select(X, 1, 0).unsqueeze(1)
            X_lys = model.globalattention(X_lys,X,X,valid_lens) + X_lys
            X_lys = model.layernorm4(X_lys)
            
            embeddings += model.linear_relu_stack[:4](X_lys).squeeze(1).tolist()
    return np.array(embeddings)

def GetRepresentation_mlp(dataloader, model, device):
    embeddings = list()
    for X, y in dataloader:
        X = X.to(torch.float32).to(device)
        
        model.eval()
        with torch.no_grad():
            embeddings += model.linear_relu_stack[:-3](X).squeeze(1).tolist()
    return np.array(embeddings)

class Visualise_t_SNE(object):
    def __init__(self, df_train, df_test, labels_train, labels_test, emb_train, emb_test):
        self.fig_test = None
        self.fig_train = None
        
        # t-SNE
        self.emb_train = TSNE(n_components=2, learning_rate='auto', init='random', random_state=43).fit_transform(emb_train)
        self.emb_test = TSNE(n_components=2, learning_rate='auto', init='random', random_state=43).fit_transform(emb_test)
        
        self.labels_train = np.array(labels_train)
        self.labels_test = np.array(labels_test)
        
        # carbamate sites identified by the research group at Durham University
        self.exp_dict = {'P15531':[12], 'P0ACF0':[67],'P22392':[12],'P39207':[9],'P0A6F5':[34],'P02925':[45],'P02925':[285],
                         'O03042':[183],'P0A853':[121],'P21276':[48],'P21276':[208],'Q9SMU8':[262],'Q9SMU8':[268],'Q9XFT3':[110],
                         'Q41932':[116],'Q42589':[65],'Q42589':[115],'Q54714':[36],'B1XLQ5':[273],'Q55085':[394],'B1XIR5':[374],
                         'B1XP79':[346],'B1XM87':[50],'B1XK41':[484],'P29033':[108],'P29033':[112],'P29033':[116],'P29033':[125],
                         'P22393':[12],'P16152':[239],'P34931':[332],'P47897':[673],'Q9H6T3':[243],'PDB_1UBQ':[6],'PDB_1UBQ':[33],
                         'PDB_1UBQ':[48],'PDB_1UBQ':[63]}
        
        # confident lys-co2 binding site by King et al.
        self.king_dict = {'P0AC33': [99],'P0A9L8': [262],'P0A8M6': [5],'P46122': [71],'P0ACD4': [124],'P46474': [737],
                          'P0AA37':[77],'P37440': [74],'P0AA95': [99],'P77257': [325],'P52696': [123],'P76044': [23],
                          'A8DYQ1': [20],'P39407': [193],'Q6XEC0': [73],'P54205': [196],'P73922': [142],'Q01951': [27],
                          'Q55247': [90],'P73853': [149],'P10549': [177],'Q01952': [53]}
        
        """
        test set gathering-------------------------------------------------
        
        """
        self.text_test_pos = list()
        self.text_test_neg = list()
        
        self.unseen_test = [-99]*len(df_test) # difficult positives
        self.text_unseen_test = list()
        
        self.exp_test = [-99]*len(df_test) # unpublished positives
        self.text_exp_test = list()
        
        self.king_test = [-99]*len(df_test) # positives by King et al.
        self.text_king_test = list()
        
        """
        training set gathering---------------------------------------------
        
        """
        self.text_train_pos = list()
        self.text_train_neg = list()
        
        self.red_index_train = [-99]*len(df_train) # minority positives
        self.text_red_index_train = list()
        
        self.exp_train = [-99]*len(df_train) # unpublished positives
        self.text_exp_train = list()
        
        self.king_train = [-99]*len(df_train) # positives by King et al.
        self.text_king_train = list()
        
        for idx, row in df_test.iterrows():
            u = row['Uniprot Entry']
            r = row['Resid']
            label = row['Label']
            text = "Uniprot: %s<br>PDB: %s<br>Resid: %s<br>Chain: %s"%(row["Uniprot Entry"],
                                row['PDB Code'].split('/')[-1], str(row['Resid']), row['Chain'])
            if label == 1:
                self.text_test_pos.append(text)
                # check if the lysine is in the training set
                df_query = df_train[(df_train['Uniprot Entry']==u)&(df_train['Resid']==r)]
                num = len(df_query)
                if num == 0:
                    self.unseen_test[idx] = 99
                    self.text_unseen_test.append(text)
                # check if the lysine is one of the unpublished ones    
                if u in list(self.exp_dict.keys()):
                    if r in self.exp_dict[u]:
                        self.exp_test[idx] = 99
                        self.text_exp_test.append(text)
                # check if the lysine is published by King et al.
                if u in list(self.king_dict.keys()):
                    if r in self.king_dict[u]:
                        self.king_test[idx] = 99
                        self.text_king_test.append(text)
            else:
                self.text_test_neg.append(text)
            
       
        for idx, row in df_train.iterrows():
            u = row['Uniprot Entry']
            r = row['Resid']
            label = row['Label']
            text = "Uniprot: %s<br>PDB: %s<br>Resid: %s<br>Chain: %s"%(row["Uniprot Entry"],
                                row['PDB Code'].split('/')[-1], str(row['Resid']), row['Chain'])
            if label == 1:
                self.text_train_pos.append(text)
                # check if the lysine has less than 3 conformations in the training set
                df_query = df_train[(df_train['Uniprot Entry']==u)&(df_train['Resid']==r)]
                num = len(df_query)
                if num <= 3:
                    self.red_index_train[idx] = 99  
                    self.text_red_index_train.append(text)
                
                if u in list(self.exp_dict.keys()):
                    if r in self.exp_dict[u]:
                        self.exp_train[idx] = 99
                        self.text_exp_train.append(text)
                
                if u in list(self.king_dict.keys()):
                    if r in self.king_dict[u]:
                        self.king_train[idx] = 99
                        self.text_king_train.append(text)
            else:
                self.text_train_neg.append(text)
        
    def show_test(self,title=None, opacity_bg=0.8, opacity_other=0.8, opacity_difficult=1, size=12, preds=None, trues=None):
        trace_specs = [[self.emb_test, self.labels_test, 0, 'Negatives', self.text_test_neg, 
                            dict(color='Pink', line=dict(color='Orange', width=2))],
                      [self.emb_test, self.labels_test, 1, 'Positives', self.text_test_pos, 
                            dict(color='LightSkyBlue', line=dict(color='MediumPurple', width=2))]]

        self.fig_test = go.Figure(data = [
            go.Scatter(
            x = X[y==label, 0], y=X[y==label, 1],
            name = name,
            text = text,
            opacity = opacity_bg,
            mode = 'markers', marker=marker
            )
            for X, y, label, name, text, marker in trace_specs
            ])
        
        # if preds and trues were supplied, plot mistakes
        try:
            self.fig_test.add_trace(
                go.Scatter(
                x = self.emb_test[np.array(preds) != np.array(trues), :][:, 0], y = self.emb_test[np.array(preds) != np.array(trues), :][:, 1],
                name = 'Mistakes',
                opacity = opacity_other,
                mode = 'markers', marker_symbol = 'x', marker=dict(color='Red', size=size, line=dict(color='Red', width=0.5))
                )
                )
        except:
            pass

        self.fig_test.add_trace(
            go.Scatter(
            x = self.emb_test[np.array(self.exp_test) == 99, :][:, 0], y = self.emb_test[np.array(self.exp_test) == 99, :][:, 1],
            name = 'Unpublished Positives',
            opacity = opacity_other,
            text = self.text_exp_test,
            mode = 'markers', marker_symbol = 'x', marker=dict(color='Grey', size=size, line=dict(color='Black', width=0.5))
            )
            )
        
        self.fig_test.add_trace(
            go.Scatter(
            x = self.emb_test[np.array(self.king_test) == 99, :][:, 0], y = self.emb_test[np.array(self.king_test) == 99, :][:, 1],
            name = 'Positives from King et al.',
            opacity = opacity_other,
            text = self.text_king_test,
            mode = 'markers', marker_symbol = 'x', marker=dict(color='#19D3F3', size=size, line=dict(color='Black', width=0.5))
            )
            )
        
        self.fig_test.add_trace(
            go.Scatter(
            x = self.emb_test[np.array(self.unseen_test) == 99, :][:, 0], y = self.emb_test[np.array(self.unseen_test) == 99, :][:, 1],
            name = 'Difficult Positives',
            opacity = opacity_difficult,
            text = self.text_unseen_test,
            mode = 'markers', marker_symbol = 'star', marker=dict(color='Lime',size=size, line=dict(color='DarkGreen', width=0.5))
            )
            )
        if title:
            self.fig_test.update_layout(title={'text': title},title_x=0.5)
        
        self.fig_test.update_xaxes(showgrid=True,showline=True, linewidth=2, linecolor='black', mirror=True)
        self.fig_test.update_yaxes(showgrid=True,showline=True, linewidth=2, linecolor='black', mirror=True)
        self.fig_test.update_layout(xaxis_title="t-SNE Dim 1", yaxis_title="t-SNE Dim 2",
                                    legend=dict(yanchor="bottom",y=0,
                                                xanchor="right",x=1,
                                                bgcolor="rgba(0,0,0,0)",))
        return self.fig_test
    
    def show_train(self, title=None, opacity_bg=0.8, opacity_other=0.8, size=12):
        trace_specs = [[self.emb_train, self.labels_train, 0, 'Negatives', self.text_train_neg, 
                            dict(color='Pink', line=dict(color='Orange', width=2))],
                      [self.emb_train, self.labels_train, 1, 'Positives', self.text_train_pos, 
                            dict(color='LightSkyBlue', line=dict(color='MediumPurple', width=2))]]
        
        self.fig_train = go.Figure(data = [
            go.Scatter(
            x = X[y==label, 0], y=X[y==label, 1],
            name = name,
            text = text,
            opacity=opacity_bg,
            mode = 'markers', marker=marker
            )
            for X, y, label, name, text, marker in trace_specs
            ])

        self.fig_train.add_trace(
            go.Scatter(
            x = self.emb_train[np.array(self.red_index_train) == 99, :][:, 0], y = self.emb_train[np.array(self.red_index_train) == 99, :][:, 1],
            name = 'Minority Positives',
            opacity=opacity_other,
            text = self.text_red_index_train,
            mode = 'markers', marker_symbol = 'star', marker=dict(color='lightsalmon', size=size, line=dict(color='Red', width=0.5))
            )
            )

        self.fig_train.add_trace(
            go.Scatter(
            x = self.emb_train[np.array(self.exp_train) == 99, :][:, 0], y = self.emb_train[np.array(self.exp_train) == 99, :][:, 1],
            name = 'Unpublished Positives',
            opacity=opacity_other,
            text = self.text_exp_train,
            mode = 'markers', marker_symbol = 'x', marker=dict(color='Grey', size=size, line=dict(color='Black', width=0.5)
            )
            )
            )
        
        self.fig_train.add_trace(
            go.Scatter(
            x = self.emb_train[np.array(self.king_train) == 99, :][:, 0], y = self.emb_train[np.array(self.king_train) == 99, :][:, 1],
            name = 'Positives from King et al.',
            opacity=opacity_other,
            text = self.text_king_train,
            mode = 'markers', marker_symbol = 'x', marker=dict(color='#19D3F3', size=size, line=dict(color='Black', width=0.5)
            )
            )
            )
        if title:
            self.fig_train.update_layout(title={'text': title},title_x=0.5)
        
        self.fig_train.update_xaxes(showgrid=True,showline=True, linewidth=2, linecolor='black', mirror=True)
        self.fig_train.update_yaxes(showgrid=True,showline=True, linewidth=2, linecolor='black', mirror=True)
        self.fig_train.update_layout(xaxis_title="t-SNE Dim 1", yaxis_title="t-SNE Dim 2",
                                    legend=dict(yanchor="top", y=1, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)",))
        return self.fig_train