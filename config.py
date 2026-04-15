class PepRLGenConfig:
    embd_pdrop = 0.3
    resid_pdrop = 0.3
    attn_pdrop = 0.3
    def __init__(self, vocab_size, block_size, is_pretrain=False,**kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.is_pretrain = is_pretrain
        for k,v in kwargs.items():
            setattr(self, k, v)

class TrainerConfig:
    batch_size = 64
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1
    lr_decay = False
    warmup_tokens = 375e6
    final_tokens = 260e9
    ckpt_path = None
    num_workers = 4
    vocab_size = 23
    epoch_rcd =[]
    train_l_rcd=[]
    train_la_rcd=[]
    train_lb_rcd=[]
    train_lc_rcd=[]
    val_l_rcd=[]
    val_la_rcd=[]
    val_lb_rcd=[]
    val_lc_rcd=[]
    avs_l_rcd=[]
    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            setattr(self, k, v)

class myconfig:
    def __init__(self, n_head=8, n_embd=64, attn_pdrop=0.2, resid_pdrop=0.2, embd_pdrop = 0.2,**kwargs):
        self.n_embd = n_embd
        self.attn_pdrop = attn_pdrop
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.n_head = n_head
        self.vocab_size= 23
        self.block_size = 14
        self.n_layer = 1
        self.weight_decay = 0.0
        self.ckpt_path = './model/TRLP_score_model.pt'
        for k, v in kwargs.items():
            setattr(self, k, v)