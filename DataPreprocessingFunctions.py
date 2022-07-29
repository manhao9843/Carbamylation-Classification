import pandas as pd
import numpy as np
import concurrent.futures
import biobox as bb
import torch
from torch import nn
import math

def distance_L2(x1, x2):
    a1, a2 = np.array(x1), np.array(x2)
    d2 = 0
    for i,j in zip(a1, a2):
        d2 += (i - j)**2
    return np.sqrt(d2)


# positional encoding
def PE(x, L):
    encoded = list()
    for i in range(L):
        arg = (2**i)*(math.pi)*x
        encoded.append(math.sin(arg))
        encoded.append(math.cos(arg))
    return encoded

def who_is_nearby(pdb_file, resid, chain, d):
    
    # identify all the CA of all the amino acids
    M = bb.Molecule()
    M.import_pdb(pdb_file)
    pos, idx = M.atomselect(chain, "*", ["CA"], get_index=True)
    
    # identify the NZ of the lysine
    center = M.atomselect(chain, [resid], ["NZ"], get_index=False)[0]
    
    # idenfity amino acids within 4 to 20 from the center NZ
    neighbours = list()
    for coordinates, index in zip(pos,idx):
        try:
            distance = distance_L2(center, coordinates)
            centered_coordinates = coordinates - center
            if distance <= d:
                pdb_file_index = index + 1
                read_file = open(pdb_file)
                for line in read_file:
                    if line[:4] != 'ATOM':
                        continue
                    if str(pdb_file_index) == line[:30].split()[1]:
                        try:
                            resid_id = int(line[:30].split()[-1])
                            amino = line[:30].split()[-3]
                        except:
                            resid_id = int(line[:30].split()[-1][1:])
                            amino = line[:30].split()[-2]
                        if resid_id != resid:
                            neighbours.append((amino, resid_id, centered_coordinates))
                read_file.close()
        except Exception as e:
            print('PDB:', pdb_file, 'Chain:',chain)
            print(e)
    return neighbours

class loader(object):
    def __init__(self, df, d):
        df = df.reset_index(drop=True)
        self.df = df
        self.x0 = range(len(df))
        self.x1 = [i+'.pdb' for i in df['PDB Code'].tolist()]
        self.x2 = df['Resid'].tolist()
        self.x3 = df['Chain'].tolist()
        self.x4 = [d] * len(df)
        self.dict = dict()
        
    def loading(self,x): # x = [idx, pdb_file, resid, chain, dmin, dmax]
        data = who_is_nearby(x[1], x[2], x[3], x[4])
        self.dict[x[0]] = data
        print(f"Row {x[0]} Processed.")
    
    def all_at_once(self):
        inputs = list(zip(self.x0, self.x1, self.x2, self.x3, self.x4))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.loading, inputs)
        self.dict = dict(sorted(self.dict.items(), key=lambda x: x[0]))

        
class OneHotEncoding(object):
    def __init__(self, L):
        self.amino_list = ['ALA','ARG','ASN','ASP','CYS','GLU','GLN','GLY','HIS','ILE','LEU','LYS','MET','PHE','PRO','SER', 'THR','TRP','TYR','VAL']
        self.df_index = list(L.dict.keys())
        self.amino_to_int = dict(zip(self.amino_list, range(len(self.amino_list))))
        self.int_to_amino = {v: k for k, v in self.amino_to_int.items()}
        self.data_transformed = list()
        self.labels = list()
        for idx, data in L.dict.items():
            sample = list()
            lys = [11] + PE(0,4) + PE(0,4) + PE(0,4)
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
                encoded = [to_int] + PE(x,4) + PE(y,4) + PE(z,4)
                sample.append(encoded)
            self.data_transformed.append(sample)
        self.encoded_list = list()
    
    def Encoding(self):
        for vectors in self.data_transformed:
            to_encode = list()
            vectors = torch.tensor(vectors)
            head = vectors[:,0].to(torch.int64)
            encoded = torch.nn.functional.one_hot(head, num_classes=len(self.amino_list))
            tail = vectors[:,1:]
            result = torch.cat((encoded,tail), 1)
            self.encoded_list.append(result)
        return 'Encoding Successful.'

    
"""
the following codes are for the 22-feature model

"""

def who_is_nearby_counts(pdb_file, resid, chain, d):
    amino_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                 'SER', 'THR', 'TRP', 'TYR', 'VAL']
    # idenfity the center
    M = bb.Molecule()
    M.import_pdb(pdb_file)
    pos, idx = M.atomselect(chain, "*", ["CA"], get_index=True)
    center = M.atomselect(chain, [resid], ["NZ"], get_index=False)[0]
    
    # counting amino acids
    freq_dict = dict(zip(amino_list, [0]*len(amino_list)))
    
    for coordinates, index in zip(pos,idx):
        try:
            distance = distance_L2(center, coordinates)
            if distance <= d:
                pdb_file_index = index + 1
                read_file = open(pdb_file)
                for line in read_file:
                    if line[:4] != 'ATOM':
                        continue
                    if str(pdb_file_index) == line[:30].split()[1]:
                        try:
                            resid_id = int(line[:30].split()[-1])
                            amino = line[:30].split()[-3]
                        except:
                            resid_id = int(line[:30].split()[-1][1:])
                            amino = line[:30].split()[-2]
                        if resid_id != resid:
                            freq_dict[amino] += 1
                read_file.close()
        except Exception as e:
            print(e)
    
    return freq_dict


class loader_counts(object):
    def __init__(self, df, d):
        self.df = df
        self.amino_list = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
                'GLU', 'GLN', 'GLY', 'HIS', 'ILE',
                'LEU', 'LYS', 'MET', 'PHE', 'PRO',
                 'SER', 'THR', 'TRP', 'TYR', 'VAL']
        for amino in self.amino_list:
            self.df[amino] = [0] * len(self.df)
        
        self.x0 = range(len(df))
        self.x1 = [i+'.pdb' for i in df['PDB Code'].tolist()]
        self.x2 = df['Resid'].tolist()
        self.x3 = df['Chain'].tolist()
        self.x4 = [d] * len(df)
        
    def loading_row(self,x): # x = [idx, pdb_file, resid, chain, d]
        freq_dict = who_is_nearby(x[1], x[2], x[3], x[4])
        for amino, FREQ in freq_dict.items():
            self.df.loc[x[0], amino] = FREQ
    
    def all_at_once(self):
        inputs = list(zip(self.x0, self.x1, self.x2, self.x3, self.x4))
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(self.loading_row, inputs)