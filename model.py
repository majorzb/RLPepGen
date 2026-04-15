import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
import matplotlib.pyplot as plt

class TransformerFFN(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1, activation="relu"):

        super(TransformerFFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout_rate = dropout

        # 网络层
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class PositionalEncoding(nn.Module):
    """Transformer位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :, :]
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)


        self.dropout = nn.Dropout(dropout)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> tuple:
        batch_size, seq_len_q, _ = q.size()
        batch_size, seq_len_k, _ = k.size()
        batch_size, seq_len_v, _ = v.size()
        # 添加激活函数检查
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        # 检查中间激活值
        if torch.isnan(Q).any() or torch.isinf(Q).any():
            print("NaN/Inf in Q")

        Q = Q.view(batch_size, seq_len_q, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len_v, self.n_heads, self.d_k).transpose(1, 2)
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)#-1e9)

        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重
        output = torch.matmul(attn_weights, V)

        # 重新排列并合并头部
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)

        # 最终线性投影
        output = self.w_o(output)
        return output, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model: int, n_heads: int, vocab_size: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> tuple:
        # 自注意力 + 残差
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # 前馈 + 残差
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.norm2(x)

        return x, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        #self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
            )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        martrix = torch.tril(
            torch.ones(315, 315)).view(1, 1, 315, 315)
        martrix[:, :, 16:, :] = 1
        martrix[:, :, :, 16:] = 1
        self.register_buffer("mask", martrix)
    def forward(self, x: torch.Tensor) -> tuple:

        y = self.norm1(x)
        self_attn_out, self_attn_weights = self.self_attn(y, y, y, self.mask)
        x = x + self_attn_out
        # 前馈
        y = self.norm4(x)
        ffn_out = self.ffn(y)
        x = x + ffn_out

        return x, self_attn_weights


class PeptideTransformer(nn.Module):
    """多肽生成Transformer模型"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size = self.config.vocab_size

        # 嵌入层
        self.src_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.rec_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # 位置编码
        self.src_pos_encoding = PositionalEncoding(config.n_embd, config.block_size)
        self.rec_pos_encoding = PositionalEncoding(config.n_embd, config.max_rec_len)

        # 编码器
        self.src_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.n_embd, config.n_head, config.embd_pdrop)
            for _ in range(config.n_layer)
        ])
        self.rec_encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(config.n_embd, config.n_head, config.embd_pdrop)
            for _ in range(config.n_layer)
        ])

        # 解码器
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(config.n_embd, config.n_head, config.embd_pdrop)
            for _ in range(config.n_layer)
        ])

        # 输出层
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.output_layer = nn.Linear(config.n_embd, config.vocab_size)
        self.linein2 = nn.Linear(config.vocab_size, 1, bias=True)

        # Dropout
        self.dropout = nn.Dropout(config.embd_pdrop)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def create_rec_mask(self, src_tokens, pad_token_id=0):
        """创建源序列mask（用于padding）"""
        src_mask = (src_tokens != pad_token_id).unsqueeze(1).unsqueeze(2)
        return src_mask

    def create_src_mask(self, tgt_tokens, pad_token_id=0):
        """创建目标序列mask（因果掩码 + padding掩码）"""
        batch_size, tgt_len = tgt_tokens.shape

        # Padding mask
        tgt_pad_mask = (tgt_tokens != pad_token_id).unsqueeze(1).unsqueeze(3)

        # Causal mask（防止看到未来信息）
        causal_mask = torch.tril(torch.ones(tgt_len, tgt_len)).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

        # 结合两种mask
        tgt_mask = tgt_pad_mask & causal_mask.to(tgt_tokens.device)
        return tgt_mask

    def freeze_first_three_layers(self):
        """只训练最后一层"""
        # 冻结前N-1层
        for i in range(self.config.n_layer):  # 索引 0, 1, 2
            for param in self.decoder_layers[i].parameters():
                param.requires_grad = False

            for param in self.src_encoder_layers[i].parameters():
                param.requires_grad = False
            for param in self.rec_encoder_layers[i].parameters():
                param.requires_grad = False

        for param in self.src_embedding.parameters():
                param.requires_grad = False
        for param in self.rec_embedding.parameters():
                param.requires_grad =False
        for name,param in self.ln_f.named_parameters():
            # 检查是否是我们要冻结的层的权重
            if name.startswith(name) and name.endswith('weight'):
                    param.requires_grad = False
            elif name.startswith(name) and name.endswith('bias'):
                    param.requires_grad = True
        for name,param in self.output_layer.named_parameters():
            # 检查是否是我们要冻结的层的权重
            if name.startswith(name) and name.endswith('weight'):
                param.requires_grad = False
            elif name.startswith(name) and name.endswith('bias'):
                param.requires_grad = True
        for name,param in self.linein2.named_parameters():
            # 检查是否是我们要冻结的层的权重
            if name.startswith(name) and name.endswith('weight'):
                param.requires_grad = False
            elif name.startswith(name) and name.endswith('bias'):
                param.requires_grad = True
        # 确保最后一层（索引3）可训练
        for name,param in self.decoder_layers[self.config.n_layer-1].named_parameters():
            # 检查是否是我们要冻结的层的权重
            if name.startswith(name) and name.endswith('weight'):
                param.requires_grad = False
            elif name.startswith(name) and name.endswith('bias'):
                param.requires_grad = True
        print("已冻结 Decoder 的前3层，只训练第4层")

    def configure_optimizers(self, train_config):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Embedding)
        blacklist_weight_modules = (torch.nn.LayerNorm)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn

                if pn.endswith('bias') or ('bias' in pn):
                    no_decay.add(fpn)
                elif (pn.endswith('weight') or ('weight' in pn)) and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer


    def forward(self, src: torch.Tensor, target: torch.Tensor = None, rec: torch.Tensor = None, p: torch.Tensor = None,
                valid_len=None, pad_idx: int = 2, is_mi: bool = True) -> Tuple:
        """前向传播"""
        batch_size, long = src.size()
        if long < 15:
            src = torch.cat([src, torch.zeros(batch_size, (15 - long), dtype=src.dtype, device=src.device)], 1)

        # 创建mask
        rec_mask = self.create_rec_mask(rec, pad_idx) #防止填充符（'<'）参与自注意力计算
        src_mask = self.create_src_mask(src, pad_idx)

        #嵌入
        src_emb = self.src_embedding(src)
        src_emb = self.src_pos_encoding(src_emb)
        src_emb = self.dropout(src_emb)
        self_attentions = []
        for i, layer in enumerate(self.src_encoder_layers):
            src_decoder, self_attn_src = layer(src_emb, src_mask)
            self_attentions.append(self_attn_src)

        rec_emb = self.rec_embedding(rec)
        rec_emb = self.rec_pos_encoding(rec_emb)
        rec_emb = self.dropout(rec_emb)
        self_attentions_rec = []
        for i, layer in enumerate(self.rec_encoder_layers):
            rec_decoder, self_attn_rec = layer(rec_emb, rec_mask)
            self_attentions_rec.append(self_attn_rec)


        encoder_output = torch.cat((src_decoder,rec_decoder), 1)
        cross_attentions = []

        # 解码
        for i, layer in enumerate(self.decoder_layers):
            decoder_output, self_attn = layer(
                encoder_output
            )
            cross_attentions.append(self_attn)
            if torch.isnan(decoder_output).any():
                print(f"NaN in decoder layer {i}")

        # 输出预测
        logits = self.output_layer(self.ln_f(decoder_output))
        seq = logits[:, :15, :]
        out = logits[:, :15, :]

        loss = None
        alpha = 1
        if target is not None:
            loss = alpha*F.cross_entropy(out.reshape(-1, out.size(-1)),target.reshape(-1))
        # 返回注意力权重用于分析
        attention_dict = {
            'encoder_attentions': self_attentions_rec,
            'decoder_self_attentions': self_attentions,
            'decoder_cross_attentions': cross_attentions
        }

        return seq, loss, attention_dict