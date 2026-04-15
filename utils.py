import random

import torch
from torch.nn import functional as F
import math
from sklearn.cluster import MiniBatchKMeans

import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def RL_sample(model, rec, batch_size, steps,begin=0, temperature=1.0, sample=False, top_k=None):
    x = torch.tensor(begin).repeat(batch_size).reshape(batch_size,1).to(next(model.parameters()).device)
    model.eval()
    miec=None
    nllloss=None
    log_probs_all = []
    probs_all = []
    for k in range(steps+1):
        if temperature<1:
            temp = float(min(0.5*(1-math.cos(math.pi*float(k/steps))+2*temperature),1.0))
        else:
            temp =1
        if k == 0:
            logits,_,_ = model(x,rec=rec.to(next(model.parameters()).device))
        elif k == 15:
            logits,_,_ = model(x[:,:-1],rec=rec.to(next(model.parameters()).device))
            continue
        else:
            logits,_,_=model(x,rec=rec.to(next(model.parameters()).device))
        logits = logits[:, k, :] / temp
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        probs_all.append(probs)
        log_probs = F.log_softmax(logits)
        log_probs_all.append(log_probs)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)
    logits,_,_=model(x[:,:-1],rec=rec.to(next(model.parameters()).device))
    targets=x[:,1:]
    log_probs = F.log_softmax(logits, -1)
    nllloss = F.nll_loss(
        log_probs.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction='none'
    ).reshape(batch_size, -1).sum(-1)
    #nllloss=F.nll_loss(torch.log((F.softmax(logits,-1).reshape(-1, logits.size(-1)))), targets.reshape(-1),reduction='none').reshape(batch_size,-1).sum(-1)
    return x,nllloss,log_probs


class EarlyStopping:
    def __init__(self, config, patience=15, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): 验证集损失不再改善的等待周期
            verbose (bool): 是否打印早停信息
            delta (float): 认为有改善的最小变化量
            path (str): 模型保存路径
            trace_func (function): 日志打印函数
        """
        self.config = config
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.tokens = None
        self.epoch = 0
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss,model,self.epoch)
            self.epoch += 1
        elif score < self.best_score + self.delta :
            self.counter += 1
            #self.save_checkpoint(val_loss, model, self.epoch)
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                self.epoch += 1
            if self.counter >= self.patience:
                self.early_stop = True
                self.epoch += 1
        elif score > self.best_score:
            self.best_score = score
            self.save_checkpoint(val_loss, model,self.epoch)
            self.counter = 0
            self.epoch += 1

    def save_checkpoint(self, val_loss, model,epoch,is_PCC=False):
        '''保存模型当验证集损失减少时'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        raw_model = model.module if hasattr(model, "module") else model
        torch.save(raw_model.state_dict(),f'{self.config.ckpt_path}_{epoch}.pt')#self.config.ckpt_path)
        self.val_loss_min = val_loss

class Experience(object):
    def __init__(self, max_size=100,reverse=False):
        #memory ->(cluster,seq,score,nllloss,miec)
        self.memory = []
        self.max_size = max_size
        # self.voc = voc
        self.reverse =reverse
    def add_experience(self, experience):
        self.memory.extend(experience)
        if len(self.memory) > self.max_size:
            # Remove duplicates
            idxs, seq = [], []
            for i, exp in enumerate(self.memory):
                if exp[0] not in seq:
                    idxs.append(i)
                    seq.append(exp[0])
            self.memory = [self.memory[idx] for idx in idxs]
            # Retain highest scores
            self.memory.sort(key=lambda x: x[-2],reverse=self.reverse)
            self.memory = self.memory[:self.max_size]
    def __len__(self):
        return len(self.memory)

    def sample(self, n):
        """Sample a batch size n of experience"""
        if len(self.memory) < n:
            raise IndexError('Size of memory ({}) is less than requested sample ({})'.format(len(self), n))
        else:
            scores = [x[1] for x in self.memory]
            probabilities = (1 - np.array(scores).squeeze()) / np.sum(1 - np.array(scores))
            sample = np.random.choice(len(self), size=n, replace=False, p=probabilities)
            sample = [self.memory[i] for i in sample]
            seq = [x[0] for x in sample]
            scores = [x[1] for x in sample]
            prior_likelihood = [x[2] for x in sample]
        return torch.tensor(seq).reshape(n,-1), torch.tensor(scores).reshape(n,-1), torch.tensor(prior_likelihood).reshape(n,-1)

def streaming_sequence_split(sequences, test_size=0.2, batch_size=5000, n_clusters=100):
    """
    流式处理大规模序列数据
    """

    # 1. 简单的序列特征提取函数（内存友好）
    def extract_simple_features(seq):
        """提取简单的序列特征，避免内存爆炸"""
        # 氨基酸组成（20维）
        aa_counts = np.zeros(22)
        aa_index = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWYXZ")}

        for aa in seq:
            if aa in aa_index:
                aa_counts[aa_index[aa]] += 1

        # 归一化
        if len(seq) > 0:
            aa_counts = aa_counts / len(seq)

        # 添加序列长度特征（归一化）
        length_feature = min(len(seq) / 1000, 1.0)  # 假设最大长度1000

        return np.concatenate([aa_counts, [length_feature]])

    # 2. 分批提取特征并在线学习聚类
    print("开始流式特征提取和聚类...")

    all_features = []
    all_indices = []

    for batch_start in range(0, len(sequences), batch_size):
        batch_end = min(batch_start + batch_size, len(sequences))
        batch_sequences = sequences[batch_start:batch_end]
        batch_indices = list(range(batch_start, batch_end))

        # 提取特征
        batch_features = np.array([extract_simple_features(seq) for seq in batch_sequences])

        all_features.append(batch_features)
        all_indices.extend(batch_indices)

        print(f"已处理: {batch_end}/{len(sequences)} 条序列")

    # 合并所有特征
    X = np.vstack(all_features)

    # 3. 使用MiniBatchKMeans进行聚类
    print("进行MiniBatchKMeans聚类...")
    kmeans = MiniBatchKMeans(
        n_clusters=min(n_clusters, len(sequences) // 100),  # 自适应聚类数
        batch_size=1000,
        random_state=42,
        n_init=3
    )
    clusters = kmeans.fit_predict(X)

    # 4. 按聚类划分
    train_indices, test_indices = [], []

    unique_clusters = np.unique(clusters)
    np.random.shuffle(unique_clusters)
    print(f"共得到 {len(unique_clusters)} 个聚类")

    for cluster_id in unique_clusters:

        cluster_mask = clusters == cluster_id
        cluster_indices = np.array(all_indices)[cluster_mask]

        '''if len(train_indices) <= len(sequences)*0.8 :
            train_indices.extend(cluster_indices)
        else:
            test_indices.extend(cluster_indices)'''
        n_test = max(1, int(len(cluster_indices) * test_size))
        test_idx = np.random.choice(cluster_indices, size=n_test, replace=False)
        train_idx = np.setdiff1d(cluster_indices, test_idx)

        train_indices.extend(train_idx)
        test_indices.extend(test_idx)

    return np.array(train_indices), np.array(test_indices)