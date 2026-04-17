import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import EarlyStopping
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, train_dataset, test_dataset, val_dataset, config, stoi, itos, is_pretrain=False):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.stoi = stoi
        self.itos = itos
        self.is_pretrain = is_pretrain
        self.tokens = 0
        self.writer = SummaryWriter(log_dir=f'./tensorboard/{self.config.log_name}')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)


    def check_model_freeze_status(self, model, verbose=True):
        """
        检查模型各层的冻结状态

        参数:
            model: PyTorch模型
            verbose: 是否打印详细信息
        """
        total_params = 0
        trainable_params = 0
        frozen_params = 0

        if verbose:
            print("=" * 80)
            print("模型冻结状态检查")
            print("=" * 80)

        # 按层名检查
        for name, param in model.named_parameters():
            total_params += param.numel()

            if param.requires_grad:
                trainable_params += param.numel()
                status = "可训练"
            else:
                frozen_params += param.numel()
                status = "已冻结"

            if verbose:
                print(f"{name:50} | {status:8} | 形状: {str(list(param.shape)):20} | 参数数: {param.numel():,}")

        # 汇总信息
        if verbose:
            print("=" * 80)
            print(f"总参数数: {total_params:,}")
            print(f"可训练参数: {trainable_params:,} ({trainable_params / total_params * 100:.2f}%)")
            print(f"冻结参数: {frozen_params:,} ({frozen_params / total_params * 100:.2f}%)")
            print("=" * 80)

        return {
            "total": total_params,
            "trainable": trainable_params,
            "frozen": frozen_params,
            "trainable_ratio": trainable_params / total_params if total_params > 0 else 0
        }

    def save_checkpoint(self):
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info(f"Saving checkpoint to {self.config.ckpt_path}")
        torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def _update_lr(self, optimizer, config):
        """处理学习率衰减逻辑 (Linear Warmup + Cosine Decay)"""
        if self.tokens < config.warmup_tokens:
            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
        else:
            progress = float(self.tokens - config.warmup_tokens) / float(
                max(1, config.final_tokens - config.warmup_tokens))
            lr_mult = max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress) - progress / 4))

        lr = config.learning_rate * lr_mult
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def run_epoch(self, split, epoch, optimizer=None, scaler=None):
        is_train = (split == 'train')
        is_val = (split == 'val')
        self.model.train(is_train)

        dataset = {
            'train': self.train_dataset,
            'val': self.val_dataset,
            'test': self.test_dataset
        }[split]

        loader = DataLoader(dataset, shuffle=is_train, pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.num_workers)

        losses = []
        accuracies = []
        pbar = tqdm(enumerate(loader), total=len(loader)) if is_train else enumerate(loader)

        # 核心迭代循环
        with torch.set_grad_enabled(is_train):
            for it, (x, y, p, rec) in pbar:
                x, y, p, rec = [t.to(self.device) for t in [x, y, p, rec]]

                with autocast():
                    logits, loss, _ = self.model(src=x, target=y, rec=rec, p=p)
                    loss = loss.mean()

                losses.append(loss.item())

                if is_train:
                    self.model.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_norm_clip)
                    scaler.step(optimizer)
                    scaler.update()

                    if self.config.lr_decay:
                        self.tokens += (y >= 0).sum().item()
                        lr = self._update_lr(optimizer, self.config)
                    else:
                        lr = self.config.learning_rate

                    pbar.set_description(f"Epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

                elif is_val:
                    # 向量化计算准确率 (SMI Accuracy)
                    preds = torch.argmax(logits, dim=-1)  # (B, S)
                    # 确保维度匹配
                    match = (preds[:, :y.size(1)] == y)
                    batch_acc = match.float().mean(dim=1).cpu().numpy()
                    accuracies.extend(batch_acc)

        avg_loss = float(np.mean(losses))

        if is_train:
            return avg_loss
        if is_val:
            return float(np.mean(accuracies))
        return avg_loss  # Test split returns loss

    def train(self):
        config = self.config
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        optimizer = raw_model.configure_optimizers(config)
        scaler = GradScaler()
        early_stopping = EarlyStopping(config, patience=50, verbose=True, path='best_model.pth')

        for epoch in range(config.max_epochs):
            # 1. Training Phase
            train_loss = self.run_epoch('train', epoch, optimizer, scaler)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            if self.is_pretrain: print(f"epoch: {epoch + 1}")

            # 2. Validation Phase
            val_metric = 0.0
            if self.val_dataset is not None and not self.is_pretrain:
                val_metric = self.run_epoch('val', epoch)
                self.writer.add_scalar('Metric/val_smi', val_metric, epoch)

            # 3. Testing Phase
            test_loss = 0.0
            if self.test_dataset is not None:
                test_loss = self.run_epoch('test', epoch)
                self.writer.add_scalar('Loss/test', test_loss, epoch)

            # 4. Logging & Saving Results
            config.epoch_rcd.append(epoch + 1)
            config.train_l_rcd.append(train_loss)
            config.val_l_rcd.append(test_loss)

            data_to_save = [config.epoch_rcd, config.train_l_rcd, config.val_l_rcd]
            if not self.is_pretrain:
                config.avs_l_rcd.append(val_metric)
                data_to_save.append(config.avs_l_rcd)

            pd.DataFrame(data_to_save).T.to_csv(f'./output/{config.save_name}_result.csv', index=False)

            # 5. Early Stopping (Based on test_loss as per your original logic)
            early_stopping(test_loss, raw_model)
            if early_stopping.early_stop:
                print("早停触发！")
                break

        self.writer.close()