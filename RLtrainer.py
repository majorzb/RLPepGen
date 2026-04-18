import math
import esm
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils import RL_sample, EarlyStopping, Experience

logger = logging.getLogger(__name__)

regression_path = "./esm2_t12_35M_UR50D-contact-regression.pt"
# Load ESM-2 model
esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local('esm2_t12_35M_UR50D.pt')
esm_model.to('cuda')
batch_converter = alphabet.get_batch_converter()

class Trainer:
    def __init__(self, prior, agent, score_model, config, stoi, itos, is_pretrain=False):
        self.prior = prior
        self.agent = agent
        self.score = score_model
        self.config = config
        self.stoi = stoi
        self.itos = itos

        self.device = next(agent.parameters()).device
        self.score_device = next(score_model.parameters()).device

        self.maxlist = Experience(300)
        self.batch_size = config.batch_size
        self.block_size = config.block_size
        self.tmp = config.tmp
        self.sigma = config.sigma

        self.polar_aa = ['R', 'N', 'D', 'E', 'Q', 'H', 'K', 'S', 'T', 'Y', 'C', 'X', 'Z']
        self.nonpolar_aa = ['A', 'V', 'I', 'L', 'M', 'F', 'W', 'G', 'P']

    def _calculate_penalties(self, seq_tokens):
        """计算序列惩罚：连续重复氨基酸惩罚 + 极性分布惩罚"""
        penalties = []
        polar_penalties = []
        valid_indices = []
        clean_seqs_esm = []
        clean_seqs_psipred = []

        for i, se_tensor in enumerate(seq_tokens):
            se_list = se_tensor.tolist()

            # 1. 序列合法性检查 (过滤无效生成)
            if self.stoi['&'] not in se_list: continue
            end_idx = se_list.index(self.stoi['&'])
            content = se_list[:end_idx]
            if self.stoi['!'] in content[1:] or self.stoi['<'] in content: continue
            if se_list.count(self.stoi['&']) != 1: continue

            # 2. 转换字符串
            aa_seq = ''.join([self.itos[idx] for idx in content if idx != self.stoi['!']])

            # 3. 连续氨基酸惩罚
            penalty = 0.0
            consecutive = 1
            for j in range(1, len(aa_seq)):
                if aa_seq[j] == aa_seq[j - 1]:
                    consecutive += 1
                else:
                    if 2 <= consecutive <= 5:
                        penalty += (consecutive - 1) * 0.75  # 简化原if-else逻辑
                    elif consecutive > 5:
                        penalty += 3.0
                    consecutive = 1

            # 4. 极性比例惩罚
            polar_count = sum(1 for aa in aa_seq if aa in self.polar_aa)
            total = len(aa_seq)
            polar_penalty = 0.0
            if total > 0:
                ratio = polar_count / total
                deviation = ratio - 0.5
                if 0 <= deviation <= 0.1:
                    polar_penalty = deviation * 2
                else:
                    polar_penalty = deviation * 4
                    if ratio > 0.7: polar_penalty *= 4

            valid_indices.append(i)
            clean_seqs_esm.append(''.join([self.itos[int(i)] for i in se_list]).replace('!', '').replace('&', '').replace('<','<pad>'))
            clean_seqs_psipred.append(aa_seq)
            penalties.append(penalty)
            polar_penalties.append(polar_penalty)

        return valid_indices, clean_seqs_esm, penalties, polar_penalties

    def run_epoch(self, optimizer):
        self.agent.train()
        self.prior.eval()
        config = self.config

        # 1. 采样序列
        # rec_input: (Batch, Seq_Len)
        rec_input = torch.tensor(config.rec, dtype=torch.long, device=self.device).unsqueeze(0).expand(self.batch_size,
                                                                                                       -1)

        with torch.no_grad():
            seqs, prior_nll, _ = RL_sample(
                self.agent, rec_input, self.batch_size, self.block_size,
                begin=self.stoi['!'], temperature=self.tmp, sample=True
            )

        # 2. 计算惩罚与过滤有效序列
        valid_idx, esm_seqs, p_seq, p_polar = self._calculate_penalties(seqs)
        if not valid_idx: return 0, 0

        # 仅保留有效样本
        valid_seqs = seqs[valid_idx].to(self.device)
        prior_nll = prior_nll[valid_idx].to(self.device)

        # 3. 计算 Affinity Score (Delt G)
        _, _, seq_tokens = batch_converter([(f"s_{i}", s) for i, s in enumerate(esm_seqs)])

        rec_str = ''.join(self.itos[i] for i in config.rec[:200]).replace('<', '<pad>')
        _, _, rec_tokens = batch_converter([(f"r_{i}", rec_str) for i in range(len(valid_idx))])

        seq_tokens, rec_tokens = seq_tokens.to(self.device), rec_tokens.to(self.device)

        with torch.no_grad():
            # 假设 score model 处理 enc (去除了起始/结束符的部分)
            enc = valid_seqs[:, 1:-1].clone()
            enc[enc == 1] = 2  # 原代码逻辑：将某种token替换为另一种

            affinity = self.score(
                enc.to(self.score_device),
                rec_input[:len(valid_idx), :200].to(self.score_device),
                seq_tokens.to(self.score_device),
                rec_tokens.to(self.score_device)
            )[0]
            delt_G_score = (1 + affinity.cpu().detach().squeeze() / 20)

        # 4. 计算 Agent 似然度与 RL Loss
        logits, _, _ = self.agent(src=valid_seqs[:, :-1], rec=rec_input[:len(valid_idx)])
        # logits shape: (B, S, V), targets: (B, S)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = valid_seqs[:, 1:].unsqueeze(-1)

        # 收集每个 token 的 log_prob
        agent_nll = -log_probs.gather(2, targets).squeeze(-1).sum(dim=-1)

        # Augmented Likelihood = Prior_Log_Prob + Sigma * Rewards
        # 注意：RL 中通常 Reward 越高越好，这里 NLL 是负对数，所以是加上 Sigma * Score
        reward_terms = delt_G_score.to(self.device) + \
                       torch.tensor(p_seq, device=self.device) + \
                       torch.tensor(p_polar, device=self.device)

        augmented_nll = prior_nll + self.sigma * reward_terms

        # Loss = [ln P(x)_augmented - ln P(x)_agent]^2
        rl_loss = torch.pow((augmented_nll - agent_nll), 2)

        # 5. Experience Replay (经验回放)
        loss = rl_loss.mean()
        if len(self.maxlist) > 4:
            exp_seqs, exp_score, exp_prior_nll = self.maxlist.sample(4)
            exp_seqs = exp_seqs.to(self.device)

            exp_logits, _, _ = self.agent(src=exp_seqs[:, :-1], rec=rec_input[:4])
            exp_log_probs = F.log_softmax(exp_logits, dim=-1)
            exp_targets = exp_seqs[:, 1:].unsqueeze(-1)
            exp_agent_nll = -exp_log_probs.gather(2, exp_targets).squeeze(-1).sum(dim=-1)

            exp_aug_nll = exp_prior_nll.to(self.device) + self.sigma * exp_score.to(self.device)
            exp_loss = torch.pow((exp_aug_nll.squeeze() - self.sigma / 3 - exp_agent_nll), 2)
            loss = (rl_loss.mean() + exp_loss.mean()) / 2

        # 更新经验池
        self.maxlist.add_experience(zip(valid_seqs.tolist(), delt_G_score.tolist(), prior_nll.tolist()))

        # 反向传播
        self.agent.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 1.0)
        optimizer.step()

        return loss.item(), delt_G_score.mean().item()

    def train(self):
        # 冻结 Prior 参数
        for param in self.prior.parameters():
            param.requires_grad = False

        optimizer = self.agent.configure_optimizers(self.config)
        early_stopping = EarlyStopping(self.config, patience=200, verbose=True, path='best_model.pth')

        pbar = tqdm(range(self.config.max_epochs))
        for epoch in pbar:
            train_loss, avg_score = self.run_epoch(optimizer)

            pbar.set_description(f"Epoch {epoch + 1} | Loss: {train_loss:.4f} | Score: {avg_score:.4f}")

            # 记录数据
            self.config.train_l_rcd.append(train_loss)
            self.config.train_la_rcd.append(avg_score)
            if (epoch + 1) % 5 == 0:
                df = pd.DataFrame({
                    'epoch': range(len(self.config.train_l_rcd)),
                    'loss': self.config.train_l_rcd,
                    'score': self.config.train_la_rcd
                })
                df.to_csv('./output/agent_result.csv', index=False)

            # 检查早停
            early_stopping(train_loss, self.agent)
            if early_stopping.early_stop:
                logger.info("Early stopping triggered.")
                break