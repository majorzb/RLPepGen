import argparse
import math
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
import esm

from config import PepRLGenConfig, myconfig
from model import PeptideTransformer
from utils import RL_sample
from score_model import Transformer


def set_seed(seed):
    if isinstance(seed, (tuple, list)):
        seed = random.randint(seed[0], seed[1])
    print(f"设置随机种子: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def clean_sequence(seq_str, stoi):
    """验证序列有效性并清理格式"""
    if '&' not in seq_str: return None
    end_idx = seq_str.find('&')
    content = seq_str[:end_idx]

    # 规则检查
    if '!' in seq_str: return None
    if seq_str.count('&') != 1: return None
    if '<' in content: return None

    # 移除特殊符号返回纯氨基酸序列
    return seq_str[:end_idx].replace('!', '').replace('<', '')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_name', type=str, default='agent_model')
    parser.add_argument('--prior_name', type=str, default='prior_model')
    parser.add_argument('--score_name', type=str, default='score_model')
    parser.add_argument('--is_eval', type=bool, default=False)
    parser.add_argument('--novel_check', type=bool, default=True)
    parser.add_argument('--novel_check_path', type=str, default='finetune')
    parser.add_argument('--gensize', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.is_eval:
        set_seed(42)
    else:
        set_seed((1, 10000))

    # 1. 基础配置与词汇表
    AA_set = sorted(
        ['A', 'G', 'V', 'L', 'I', 'P', 'Y', 'F', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H', '<', '!',
         '&', 'X', 'Z'])
    stoi = {ch: i for i, ch in enumerate(AA_set)}
    itos = {i: ch for i, ch in enumerate(AA_set)}

    # 受体序列处理
    raw_rec = 'TSMVSMPLYAVMYPVFNELERVNLSAAQTLRAAFIKAEKENPGLTQDIIMKILEKKSVEVNFTESLLRMAADDVEEYMIERPEPEFQDLNEKARALKQILSKIPDEINDRVRFLQTIKDIASAIKELLDTVNNVFKKYQYQNRRALEHQKKEFVKYSKSFSDTLKTYFKDGKAINVFVSANRLIHQTNLILQTFKTVA'
    padded_rec = (raw_rec + '<' * 300)[:300]
    rec_tensor = torch.tensor([stoi[c] for c in padded_rec], dtype=torch.long, device=device).unsqueeze(0).repeat(
        args.batch_size, 1)

    # 2. 模型加载
    mconf = PepRLGenConfig(len(AA_set), 15, n_layer=4, n_head=8, max_rec_len=300, n_embd=256, is_pretrain=False)

    agent = PeptideTransformer(mconf).to(device)
    agent.load_state_dict(torch.load(f'./model/{args.save_name}.pt', map_location=device))
    agent.eval()

    prior = PeptideTransformer(mconf).to(device)
    prior.load_state_dict(torch.load(f'./model/{args.prior_name}.pt', map_location=device))
    prior.eval()

    # ESM 与 Score 模型
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local('esm2_t12_35M_UR50D.pt')
    esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    score_model = Transformer(esm_model).to(device)
    score_model.load_state_dict(torch.load(f'./model/{args.score_name}.pt', map_location=device))
    score_model.eval()

    # 3. 序列采样循环
    results = []
    gen_iter = math.ceil(args.gensize / args.batch_size)

    print(f"开始生成 {args.gensize} 条序列...")
    for _ in tqdm(range(gen_iter)):
        with torch.no_grad():
            # Agent 采样
            y, agent_nll, _ = RL_sample(agent, rec_tensor, args.batch_size, 15, begin=stoi['!'], temperature=1.0,
                                        sample=True)

            # Prior 计算似然
            prior_logits, _, _ = prior(y[:, :-1], rec=rec_tensor)
            prior_nll = F.cross_entropy(prior_logits.transpose(1, 2), y[:, 1:], reduction='none').sum(dim=1)

            # 准备 Score 模型输入
            esm_seqs = []
            for row in y:
                # 转换 token 为字符串并处理 pad/special
                s = ''.join([itos[idx.item()] for idx in row[1:-1]])
                esm_seqs.append(s.replace('!', '').replace('&', '').replace('<', '<pad>'))

            _, _, seq_tokens = batch_converter([(f"s_{j}", s) for j, s in enumerate(esm_seqs)])
            rec_str = raw_rec[:200].replace('<', '<pad>')
            _, _, rec_tokens = batch_converter([(f"r_{j}", rec_str) for j in range(len(esm_seqs))])

            # 特殊 token 替换 (1 -> 2) 逻辑保持
            enc = y[:, 1:-1].clone()
            enc[enc == 1] = 2

            scores = score_model(enc.to(device), rec_tensor[:, :200], seq_tokens.to(device), rec_tokens.to(device))[0]

            # 记录本批次结果
            for j in range(len(y)):
                full_str = ''.join([itos[idx.item()] for idx in y[j][1:]])
                results.append({
                    'full_seq': full_str,
                    'agent_nll': agent_nll[j].item(),
                    'prior_nll': prior_nll[j].item(),
                    'score': scores[j].item()
                })

    df = pd.DataFrame(results)

    # 4. 评估逻辑
    # 有效性过滤
    df['clean_seq'] = df['full_seq'].apply(lambda x: clean_sequence(x, stoi))
    valid_df = df.dropna(subset=['clean_seq']).copy()

    valid_ratio = len(valid_df) / len(df['full_seq'])
    unique_df = valid_df.drop_duplicates(subset=['clean_seq'])
    uniq_ratio = len(unique_df) / len(valid_df) if len(valid_df) > 0 else 0

    # 新颖性检查
    novelty = 0
    real_dup_count = 0
    if args.novel_check:
        train_data = pd.read_csv(f'./dataset/{args.novel_check_path}.csv')
        train_seqs = set(train_data.iloc[:, 0].astype(str).tolist())

        novel_df = unique_df[~unique_df['clean_seq'].isin(train_seqs)].copy()
        real_dup_count = len(unique_df) - len(novel_df)
        novelty = len(novel_df) / len(unique_df) if len(unique_df) > 0 else 0

        # 保存结果
        novel_df.to_csv(f'./output/{args.save_name}_eval.csv', index=False)

        # TBR 计算 (Score <= -15)
        high_score_count = (novel_df['score'] <= -15).sum()
        tbr = high_score_count / len(novel_df) if len(novel_df) > 0 else 0
    else:
        novel_df = unique_df
        high_score_count = 0
        tbr = 0

    # 5. 输出报告
    report_path = f'./output/{args.save_name}_evaluation.log'
    with open(report_path, 'w', encoding='utf-8') as f:
        log_content = [
            "=== 肽序列生成评估报告 ===",
            f"有效性 (Validity): {valid_ratio:.4f}",
            f"唯一性 (Uniqueness): {uniq_ratio:.4f}",
            f"新颖性 (Novelty): {novelty:.4f}",
            f"重复序列数 (Internal Dups): {len(valid_df) - len(unique_df)}",
            f"训练集重复数 (Train Set Dups): {real_dup_count}",
            f"Agent NLL 均值: {valid_df['agent_nll'].mean():.4f}",
            f"Prior NLL 均值: {valid_df['prior_nll'].mean():.4f}",
            f"预测评分均值 (Score Avg): {valid_df['score'].mean():.4f}",
            f"高分序列数 (Score <= -15): {high_score_count}",
            f"TBR (High Score Ratio): {tbr:.4f}"
        ]
        for line in log_content:
            print(line)
            f.write(line + '\n')


if __name__ == '__main__':
    main()
