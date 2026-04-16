import pandas as pd
import argparse
import numpy as np

from config import RLPepGenConfig, TrainerConfig
from model import  RLPepGen
from trainer import Trainer
from dataload import AAdataset

import random
import torch
#from z import *

torch.cuda.empty_cache()  # 清理缓存
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if __name__ == '__main__':
    #输入参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='data',
                        help="name of the dataset to train on", required=False)
    parser.add_argument('--n_layer', type=int, default=4,
                        help="number of layers", required=False)
    parser.add_argument('--n_head', type=int, default=8,
                        help="number of heads", required=False)
    parser.add_argument('--n_embd', type=int, default=256,
                        help="embedding dimension", required=False)
    parser.add_argument('--max_epochs', type=int, default=10,
                        help="total epochs", required=False)
    parser.add_argument('--batch_size', type=int, default=128,
                        help="batch size", required=False)
    parser.add_argument('--learning_rate', type=float,
                        default=4e-5, help="learning rate", required=False)
    parser.add_argument('--pre_save_name', type=str, default='pretrain_model',
                        help="modelname", required=False)
    parser.add_argument('--is_pretrain', type=bool, default=False,
                        help="pretain", required=False)
    parser.add_argument('--save_name', type=str, default='pretrain_model',
                        help="modelname", required=False)
    parser.add_argument('--is_continue', type=bool, default=False,
                        help="properties to be used for condition", required=False)
    parser.add_argument('--continue_save_name', type=str, default='',
                        help="modelname", required=False)
    parser.add_argument('--log_name', type=bool, default='pretrain_model',
                        help="The path for saving the TensorBoard log files", required=False)
    args = parser.parse_args() #get metric
    #随机种子
    set_seed(42)
    #数据预处理
    data = pd.read_csv('./dataset/'+args.data_name + '.csv')
    data = data.sample(frac=1).reset_index(drop=True)  # randomlised
    num_data=data.shape[0]
    train_data=data.iloc[:int(num_data*0.008),:].copy().reset_index(drop=True)
    val_data=data.iloc[int(num_data*0.899):int(num_data*0.9),:].copy().reset_index(drop=True)
    test_data=data.iloc[int(num_data*0.999):,:].copy().reset_index(drop=True)
    AA_set=sorted(['A','G','V','L','I','P','Y','F','W','S','T','C','M','N','Q','D','E','K','R','H','<','!','&','X','Z'])
    clu_size = 13
    max_seq_len=14
    max_len=15
    num_delt_G=1
    stoi = { ch:i for i,ch in enumerate(AA_set) }
    seqlistAA=[]
    if args.is_pretrain:
        train_dataset=AAdataset(args,train_data,AA_set,max_seq_len,max_rec_len=300,aug_prob =0.3,delt_G=None)
        val_dataset = AAdataset(args, val_data, AA_set, max_seq_len, max_rec_len=300, aug_prob=0,delt_G=None)
        test_dataset = AAdataset(args, test_data, AA_set, max_seq_len, max_rec_len=300, aug_prob=0,delt_G=None)
    else:
        train_dataset=AAdataset(args,train_data,AA_set,max_seq_len,max_rec_len=300,aug_prob =0)
        val_dataset=AAdataset(args,val_data,AA_set, max_seq_len,max_rec_len=300,aug_prob =0)
        test_dataset=AAdataset(args,test_data,AA_set, max_seq_len,max_rec_len=300,aug_prob = 0)
    #模型定义
    mconf = RLPepGenConfig(train_dataset.vocab_size, max_len,max_rec_len=300, num_delt_G=num_delt_G,
                          n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, is_pretrain=args.is_pretrain,
                          )

    model = RLPepGen(mconf)
    if args.is_continue:
        model.load_state_dict(torch.load(f'./model/{args.pre_save_name}.pt'), strict=False)
        print('load model')
    device = torch.cuda.current_device()
    model = model.to(device)
    model.freeze_first_three_layers()

    #训练函数定义
    tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                          lr_decay=True, warmup_tokens=0.1 * len(train_data) * max_seq_len,
                          final_tokens=args.max_epochs * len(train_data) * max_len,
                          num_workers=1, ckpt_path=f'./model/{args.save_name}.pt', block_size=max_len, generate=False,
                          log_name=args.log_name,save_name=args.save_name)

    trainer = Trainer(model, train_dataset, test_dataset, val_dataset,
                      tconf, train_dataset.stoi, train_dataset.itos, is_pretrain=args.is_pretrain)
    #开始训练
    df = trainer.train()