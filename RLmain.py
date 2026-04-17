import esm
import argparse
from model import RLPepGen
from numpy import random
from RLtrainer import Trainer
from config import myconfig, RLPepGenConfig, TrainerConfig
from score_model import *

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--prior_name', type=str, default = 'prior_model', help="The prior model", required=False)
        parser.add_argument('--score_name', type=str, default = 'score_model', help="The scoring function", required=False)
        parser.add_argument('--batch_size', type=int, default = 64, help="batch size", required=False)
        parser.add_argument('--gen_size', type=int, default = 10000, help="number of times to generate from a batch", required=False)
        parser.add_argument('--vocab_size', type=int, default = 25, help="number of layers", required=False)
        parser.add_argument('--block_size', type=int, default = 15, help="number of layers", required=False)
        parser.add_argument('--n_layer', type=int, default = 4, help="number of layers", required=False)
        parser.add_argument('--n_head', type=int, default = 8, help="number of heads", required=False)
        parser.add_argument('--n_embd', type=int, default = 256, help="embedding dimension", required=False)
        parser.add_argument('--tmp', type=float, default = 1, help="temperature", required=False)
        parser.add_argument('--max_epochs', type=int, default=800,help="total epochs", required=False)
        parser.add_argument('--is_pretrain', type=bool, default=False, #args bug(bool('False')=True,被认为是str))给予任意值都是True，有需求再使用
                            help="properties to be used for condition", required=False)
        parser.add_argument('--save_name', type=str, default='agent_model2',
                            help="modelname", required=False)
        parser.add_argument('--sigma', type=int, default = 1, help="sigma", required=False)
        parser.add_argument('--learning_rate', type=float,default=4e-5, help="learning rate", required=False) #默认6e-3
        args = parser.parse_args()
        set_seed(42)
        AA_set = sorted(['A', 'G', 'V', 'L', 'I', 'P', 'Y', 'F', 'W', 'S', 'T', 'C', 'M', 'N', 'Q', 'D', 'E', 'K', 'R', 'H', '<','!', '&','X','Z'])
        stoi = {ch: i for i, ch in enumerate(AA_set)}
        itos = {i: ch for i, ch in enumerate(AA_set)}
        rec = 'TSMVSMPLYAVMYPVFNELERVNLSAAQTLRAAFIKAEKENPGLTQDIIMKILEKKSVEVNFTESLLRMAADDVEEYMIERPEPEFQDLNEKARALKQILSKIPDEINDRVRFLQTIKDIASAIKELLDTVNNVFKKYQYQNRRALEHQKKEFVKYSKSFSDTLKTYFKDGKAINVFVSANRLIHQTNLILQTFKTVA'
        rec1 = [stoi[i] for i in rec]
        rec += str('<') * (300 - len(rec))
        rec = [stoi[i] for i in rec]
        max_seq_len=14
        max_len=15

        mconf = RLPepGenConfig(args.vocab_size, args.block_size,n_layer=args.n_layer, n_head=args.n_head,max_rec_len=300,
                              n_embd=args.n_embd, is_pretrain=args.is_pretrain)
        prior = RLPepGen(mconf)
        prior.load_state_dict(torch.load('./model/'+args.prior_name+'.pt'))
        prior.to('cuda:0')
        agent = RLPepGen(mconf)

        is_continue = False
        if is_continue:
            agent.load_state_dict(torch.load('./model/' + args.save_name+'.pt'))
            print('load model from %s' % args.save_name)
        else:
            agent.load_state_dict(torch.load('./model/' + args.prior_name + '.pt'))
        agent.to('cuda:0')

        gen_iter = math.ceil(args.gen_size / args.batch_size)
        myconf = myconfig()

        regression_path = "./esm2_t12_35M_UR50D-contact-regression.pt"
        # Load ESM-2 model
        esm_model, alphabet = esm.pretrained.load_model_and_alphabet_local('esm2_t12_35M_UR50D.pt')
        esm_model.to(device)
        batch_converter = alphabet.get_batch_converter()
        esm_model.eval()
        score_model=Transformer(esm_model).to(device)
        score_model.load_state_dict(torch.load(f'./model/{args.score_name}.pt'))
        score_model.eval()
        score_model.to('cuda:0')

        tconf = TrainerConfig(max_epochs=args.max_epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,tmp = args.tmp,
                                num_workers=4, ckpt_path=f'./model/{args.save_name}.pt', block_size=max_len, generate=False,sigma=args.sigma, rec = rec)
        RLtrainer = Trainer(prior,agent,score_model,
                            tconf,stoi,itos)
        RLtrainer.train()