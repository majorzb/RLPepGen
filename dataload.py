import torch
from torch.utils.data import Dataset

import numpy as np
import random
torch.manual_seed(42)

AA_set = sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','X','Z','!','&','<'])
def randomize_aaseq(seq):
    #按性质将氨基酸分组
    groups = [
        list("AVILM"),  # 疏水/脂肪族
        list("FYW"),  # 芳香族
        list("KRH"),  # 带正电 (碱性)
        list("DE"),  # 带负电 (酸性)
        list("STNQ"),  # 极性不带电
    ]
    sub_ratio = 0.5 #每个氨基酸的突变概率
    for i in range(len(seq)):
        if random.random() < sub_ratio:
            current_aa = seq[i]
            # 寻找当前氨基酸所在的性质组
            target_group = AA_set  # 默认全局
            for group in groups:
                if current_aa in group:
                    target_group = group
                    break
            # 从同组中选一个，如果组内只有它自己，则从全局选
            if len(target_group) > 1:
                new_aa = random.choice([aa for aa in target_group if aa != current_aa])
            else:
                new_aa = random.choice([aa for aa in AA_set if aa != current_aa])
            seq = list(seq)
            seq[i] = new_aa
            seq = ''.join(seq)
    return seq

class AAdataset(Dataset):
    def __init__(self,args,data,chars,block_size,max_rec_len=None,aug_prob = 0.5,seq_idx=1,rec_idx=2,delt_G=3):
        vocab_size=len(chars)
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.max_len = block_size
        self.max_rec_len = max_rec_len
        self.vocab_size = vocab_size
        self.data = data
        self.aug_prob = aug_prob
        self.seq=np.array(data.iloc[:,:seq_idx]).copy()
        self.rec=np.array(data.iloc[:,seq_idx:rec_idx]).copy()
        if delt_G !=None:
            self.delt_G = torch.tensor(np.array(data.iloc[:, rec_idx:delt_G]), dtype=torch.float).squeeze(1)
        else:
            self.delt_G=np.zeros((len(data)))
    def __len__(self):
            return len(self.data)
    def __getitem__(self, idx):
        seq, rec, delt_G = str(self.seq[idx][0]), str(self.rec[idx][0]), self.delt_G[idx]
        p = np.random.uniform()
        if p < self.aug_prob:
            seq=randomize_aaseq(seq)
        if len(seq)<=14:
            seq = '!'+seq+'&'
        else:
            seq = '!'+seq
        seq += str('<')*(self.max_len+2 - len(seq))
        rec += str('<')*(self.max_rec_len - len(rec))
        dix =  [self.stoi[s] for s in seq]
        rec_dix = [self.stoi[s]for s in rec]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        rec_dix = torch.tensor(rec_dix, dtype=torch.long)
        return x, y, delt_G, rec_dix
    def __sample__(self, batch):
        n_sample = np.random.choice(range(0,len(self.seq)),batch)