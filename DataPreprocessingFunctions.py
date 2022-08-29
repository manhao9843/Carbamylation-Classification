import pandas as pd
import numpy as np
import concurrent.futures
import biobox as bb
import torch
from torch import nn
import math

def distance_L2(x1, x2):
    a1, a2 = np.array(x1), np.array(x2)
    d = 0
    for i,j in zip(a1, a2):
        d += (i - j)**2
    return np.sqrt(d)

def Positional_Encoding(x, L):
    encoded = list()
    for i in range(L):
        arg = (2**i)*(math.pi)*x
        encoded.append(math.sin(arg))
        encoded.append(math.cos(arg))
    return encoded

def who_is_nearby(pdb_file, resid, chain, d):
    # identify the origin
    M = bb.Molecule()
    M.import_pdb(pdb_file)
    center = M.atomselect(chain, [resid], ["NZ"], get_index=False)[0]
    
    # store amino acid info
    neighbours = list()
    
    # dict for amino acid counts
    amino_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    freq_dict = dict(zip(amino_list, [0]*len(amino_list)))
    
    # a flag indicating if a resid has already been included
    skip_resid = None
    
    read_file = open(pdb_file)
    for line in read_file:
        if line[:4] != 'ATOM':
            continue
        
        # select the right chain
        if chain != line[21].strip():
            continue
        
        # skip if the resid has already been included
        if line[22:26].strip() == skip_resid:
            continue
        
        # get coordinates
        xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        distance = distance_L2(center, xyz)
        if distance <= 12:
            xyz_centered = xyz - center
            
            # mark the resid included
            skip_resid = line[22:26].strip()
            current_resid = int(line[22:26].strip())
            
            # do not include the target lysine
            if current_resid != resid:
                amino = line[17:21].strip()
                freq_dict[amino] += 1
                resid_info = (amino, current_resid, xyz_centered)
                neighbours.append(resid_info)
    return neighbours, freq_dict

"""
apply multi-threading to extract inputs for the neural network, 
along with the amino acid counts (added to the additional 20 columns of the dataframe)
"""
class loader(object):
    def __init__(self, df, d):
        df = df.reset_index(drop=True)
        self.df = df
        self.idx = range(len(df))
        self.pdbs = [i+'.pdb' for i in df['PDB Code'].tolist()]
        self.resids = df['Resid'].tolist()
        self.chains = df['Chain'].tolist()
        self.distances = [d] * len(df)
        self.dict = dict()
        self.amino_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                           'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        for amino in self.amino_list:
            self.df[amino] = [0] * len(self.df)

    def loading(self,x): # x = [idx, pdb_file, resid, chain, d]
        neighbours, freq_dict = who_is_nearby(x[1], x[2], x[3], x[4])
        self.dict[x[0]] = neighbours
        for amino, FREQ in freq_dict.items():
            self.df.loc[x[0], amino] = FREQ
        print(f"Row {x[0]} Processed.")
    
    def all_at_once(self):
        inputs = list(zip(self.idx, self.pdbs, self.resids, self.chains, self.distances))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.loading, inputs)
        self.dict = dict(sorted(self.dict.items(), key=lambda x: x[0]))
        
class Encoder(object):
    def __init__(self, L):
        self.amino_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER', 'THR','TRP','TYR','VAL']
        self.df = L.df
        self.df_index = list(L.dict.keys())
        self.amino_to_int = dict(zip(self.amino_list, range(len(self.amino_list))))
        self.int_to_amino = {v: k for k, v in self.amino_to_int.items()}
        
        self.data_transformed = list()
        
        self.labels = list()
        
        # transform amino acids to integers, apply position encoding to coordinates
        for idx, data in L.dict.items():
            sample = list()
            lys = [11] + Positional_Encoding(0,4) + Positional_Encoding(0,4) + Positional_Encoding(0,4)
            sample.append(lys)
            label = L.df.at[idx, 'Label']
            if label == 1:
                self.labels.append(1)
            else:
                self.labels.append(0)
            for i in data:
                to_int = self.amino_to_int[i[0]]
                x = list(i[2])[0]
                y = list(i[2])[1]
                z = list(i[2])[2]
                encoded = [to_int] + Positional_Encoding(x,4) + Positional_Encoding(y,4) + Positional_Encoding(z,4)
                sample.append(encoded)
            self.data_transformed.append(sample)

        self.encoded_list = list()
    
    def encoding(self):
        # apply one-hot-encoding to the first integer
        for vectors in self.data_transformed:
            to_encode = list()
            vectors = torch.tensor(vectors)
            head = vectors[:,0].to(torch.int64)
            encoded = torch.nn.functional.one_hot(head, num_classes=len(self.amino_list))
            tail = vectors[:,1:]
            result = torch.cat((encoded,tail), 1)
            self.encoded_list.append(result)
        return 'Encoding Successful.'