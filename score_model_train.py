import math
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy.stats import pearsonr
from tqdm import tqdm
from tensorboardX import SummaryWriter
import esm

# 导入自定义模块
from score_model import Transformer
from utils import streaming_sequence_split, EarlyStopping

# --- 全局配置 ---
VOCAB_LIST = ['<cls>', '<pad>', '<eos>', '<unk>', 'L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q',
              'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O', '.', '-', '<null_1>', '<mask Westerman>']
STOI = {token: idx for idx, token in enumerate(VOCAB_LIST)}
ITOS = {idx: token for idx, token in enumerate(VOCAB_LIST)}


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# --- 数据集定义 ---
class Dataset(Dataset):
    def __init__(self, data, stoi, itos):
        self.seqs = data.iloc[:, 0].values
        self.recs = data.iloc[:, 1].values
        self.labels = data.iloc[:, 2].values.astype(np.float32)
        self.stoi = stoi
        self.itos = itos

    def __len__(self):
        return len(self.labels)

    def _pad_sequence(self, tokens, length):
        if len(tokens) < length:
            tokens.extend([self.stoi['<pad>']] * (length - len(tokens)))
        return tokens[:length]

    def __getitem__(self, idx):
        # 转换并填充序列
        seq_tokens = [self.stoi.get(a, self.stoi['<unk>']) for a in self.seqs[idx]]
        rec_tokens = [self.stoi.get(a, self.stoi['<unk>']) for a in self.recs[idx]]

        seq_pad = self._pad_sequence(seq_tokens, 14)
        rec_pad = self._pad_sequence(rec_tokens, 200)

        # 供 ESM 使用的原字符串
        seq_str = ''.join([self.itos[i] for i in seq_pad])
        rec_str = ''.join([self.itos[i] for i in rec_pad])

        return {
            "x": torch.tensor(seq_pad),
            "rec": torch.tensor(rec_pad),
            "label": torch.tensor(self.labels[idx]),
            "seq_str": seq_str,
            "rec_str": rec_str
        }


# --- 训练器类 ---
class ScoreTrainer:
    def __init__(self, model, config, batch_converter, device):
        self.model = model
        self.config = config
        self.batch_converter = batch_converter
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=4e-4)
        self.writer = SummaryWriter(log_dir=f'./tensorboard/{config.save_name}')
        self.early_stopping = EarlyStopping(config, patience=50, verbose=True, path='best_model.pth')

    def _update_learning_rate(self, it, total_iters):
        """ 余弦退火学习率更新 """
        progress = it / total_iters
        lr_mult = max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress) - progress / 4))
        lr = self.config.lr * lr_mult
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def run_epoch(self, loader, epoch, is_train=True):
        self.model.train(is_train)
        losses = []
        all_preds = []
        all_trues = []

        pbar = tqdm(loader, desc=f"Epoch {epoch} [{'Train' if is_train else 'Val'}]")

        for it, batch in enumerate(pbar):
            # 数据迁移到 GPU
            x = batch['x'].to(self.device)
            rec = batch['rec'].to(self.device)
            labels = batch['label'].to(self.device)

            # ESM Tokenization
            _, _, seq_esm = self.batch_converter([(f"s_{i}", s) for i, s in enumerate(batch['seq_str'])])
            _, _, rec_esm = self.batch_converter([(f"r_{i}", r) for i, r in enumerate(batch['rec_str'])])
            seq_esm, rec_esm = seq_esm.to(self.device), rec_esm.to(self.device)

            with torch.set_grad_enabled(is_train):
                # 前向传播
                outputs = self.model(x, rec, seq_esm, rec_esm)[0].squeeze()
                loss = F.mse_loss(outputs, labels)

                if is_train:
                    # 更新学习率 (按 iteration 更新)
                    total_steps = len(loader) * self.config.max_epoch
                    current_step = epoch * len(loader) + it
                    self._update_learning_rate(current_step, total_steps)

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
                else:
                    all_preds.append(outputs.detach().cpu().numpy())
                    all_trues.append(labels.detach().cpu().numpy())

            losses.append(loss.item())
            pbar.set_postfix(loss=np.mean(losses))

        avg_loss = np.mean(losses)
        metrics = {"loss": avg_loss}

        if not is_train:
            preds = np.concatenate(all_preds).flatten()
            trues = np.concatenate(all_trues).flatten()
            metrics["pcc"], _ = pearsonr(trues, preds)

        return metrics


# --- 主程序 ---
class Config:
    def __init__(self):
        self.lr = 4e-5
        self.max_epoch = 800
        self.batch_size = 256
        self.save_name = 'score_model'
        self.ckpt_path = f'./model/{self.save_name}.pt'


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    # 1. 加载数据
    data = pd.read_csv('dataset/finetune.csv').sample(frac=1).reset_index(drop=True)
    train_idx, test_idx = streaming_sequence_split(data.iloc[:1000, 0].values)

    train_loader = DataLoader(Dataset(data.iloc[train_idx], STOI, ITOS),
                              batch_size=config.batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(Dataset(data.iloc[test_idx], STOI, ITOS),
                             batch_size=config.batch_size, shuffle=False, pin_memory=True)

    # 2. 初始化模型
    esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local('esm2_t12_35M_UR50D.pt')
    esm_model.to(device).eval()
    batch_converter = alphabet.get_batch_converter()

    model = Transformer(esm_model).to(device)
    trainer = ScoreTrainer(model, config, batch_converter, device)

    # 3. 循环训练
    history = {"train_loss": [], "test_loss": [], "pcc": []}

    for epoch in range(config.max_epoch):
        train_metrics = trainer.run_epoch(train_loader, epoch, is_train=True)
        test_metrics = trainer.run_epoch(test_loader, epoch, is_train=False)

        # 日志记录
        trainer.writer.add_scalar('Loss/train', train_metrics['loss'], epoch)
        trainer.writer.add_scalar('Loss/test', test_metrics['loss'], epoch)
        trainer.writer.add_scalar('Metric/PCC', test_metrics['pcc'], epoch)

        history["train_loss"].append(train_metrics['loss'])
        history["test_loss"].append(test_metrics['loss'])
        history["pcc"].append(test_metrics['pcc'])

        print(
            f"Epoch {epoch} | Train Loss: {train_metrics['loss']:.4f} | Test Loss: {test_metrics['loss']:.4f} | PCC: {test_metrics['pcc']:.4f}")

        # 早停与保存
        trainer.early_stopping(test_metrics['loss'], model)
        pd.DataFrame(history).to_csv(f'./output/{config.save_name}_result.csv', index=False)

        if trainer.early_stopping.early_stop:
            print("Early stopping triggered!")
            break


if __name__ == '__main__':
    main()