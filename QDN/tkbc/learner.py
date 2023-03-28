import argparse
from typing import Dict
import logging
import torch
from torch import optim

from datasets import TemporalDataset
from optimizers import TKBCOptimizer
from models import TeLM
from regularizers import N3,  La3
import os


parser = argparse.ArgumentParser(
    description="TeLM"
)
parser.add_argument(
    '--dataset', type=str,
    help="Dataset name"
)

parser.add_argument(
    '--model', default='TeLM', type=str,
    help="Model Name"
)
parser.add_argument(
    '--max_epochs', default=200, type=int,
    help="Number of epochs."
)
parser.add_argument(
    '--valid_freq', default=5, type=int,
    help="Number of epochs between each valid."
)
parser.add_argument(
    '--rank', default=100, type=int,
    help="Factorization rank."
)
parser.add_argument(
    '--batch_size', default=1000, type=int,
    help="Batch size."
)
parser.add_argument(
    '--learning_rate', default=1e-1, type=float,
    help="Learning rate"
)
parser.add_argument(
    '--emb_reg', default=0., type=float,
    help="Embedding regularizer strength"
)
parser.add_argument(
    '--time_reg', default=0., type=float,
    help="Timestamp regularizer strength"
)
parser.add_argument(
    '--no_time_emb', default=False, action="store_true",
    help="Use a specific embedding for non temporal relations"
)
parser.add_argument(
    '--time_granularity', default=1, type=int, 
    help="Time granularity for time embeddings"
)


args = parser.parse_args()

def avg_both(mrrs: Dict[str, float], hits: Dict[str, torch.FloatTensor]):
            """
            aggregate metrics for missing lhs and rhs
            :param mrrs: d
            :param hits:
            :return:
            """
            m = (mrrs['lhs'] + mrrs['rhs']) / 2.
            h = (hits['lhs'] + hits['rhs']) / 2.
            return {'MRR': m, 'hits@[1,3,10]': h}

def learn(model=args.model,
          dataset=args.dataset,
          rank=args.rank,
          learning_rate = args.learning_rate,
          batch_size = args.batch_size, 
          emb_reg=args.emb_reg, 
          time_reg=args.time_reg,
          time_granularity=args.time_granularity,
         ):


    root = 'results/'+ dataset +'/' + model
    modelname = model
    datasetname = dataset

    ##restore model parameters and results
    PATH=os.path.join(root,'rank{:.0f}/lr{:.4f}/batch{:.0f}/time_granularity{:02d}/emb_reg{:.5f}/time_reg{:.5f}/'.format(rank,learning_rate,batch_size, time_granularity, emb_reg, time_reg))
    
    dataset = TemporalDataset(dataset)
    
    sizes = dataset.get_shape()
    model = {
        'TeLM': TeLM(sizes, rank, no_time_emb=args.no_time_emb, time_granularity=time_granularity)
    }[model]
    model = model.cuda()


    opt = optim.Adagrad(model.parameters(), lr=learning_rate)
    #opt_pretrain = optim.Adagrad(model.parameters(), lr=learning_rate)
    

    
    print("Start training process: ", modelname, "on", datasetname, "using", "rank =", rank, "lr =", learning_rate, "emb_reg =", emb_reg, "time_reg =", time_reg, "time_granularity =", time_granularity)

    emb_reg = N3(emb_reg)
    time_reg = La3(time_reg)
  
    # Results related
    try:
        os.makedirs(PATH)
    except FileExistsError:
        pass
    #os.makedirs(PATH)
    patience = 0
    mrr_std = 0

    curve = {'train': [], 'valid': [], 'test': []}

    epoch_pretrain = 50
    for epoch in range(args.max_epochs):
        print("[ Epoch:", epoch, "]")
        examples = torch.from_numpy(
            dataset.get_train().astype('int64')
        )

        model.train()

        optimizer = TKBCOptimizer(
            model, emb_reg, time_reg, opt,
            batch_size=batch_size
        )
        
        #if epoch >= epoch_pretrain:
        optimizer.epoch(examples,pre_train=False)
        #else:
            #optimizer_pretrain.epoch(examples,pre_train=True)

        
        if epoch < 0 or (epoch + 1) % args.valid_freq == 0:

            if dataset.interval: #for datasets involving time intervals, no eval() for training data
                valid, test = [
                    # avg_both(*dataset.eval(model, split, -1, use_left_queries=args.use_left))
                    avg_both(*dataset.eval(model, split, -1))
                    for split in ['valid', 'test']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])

            else:
                valid, test, train = [
                    # avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000, use_left_queries=args.use_left))
                    avg_both(*dataset.eval(model, split, -1 if split != 'train' else 50000))
                    for split in ['valid', 'test', 'train']
                ]
                print("valid: ", valid['MRR'])
                print("test: ", test['MRR'])
                print("train: ", train['MRR'])

            # Save results

            f = open(os.path.join(PATH, 'result.txt'), 'w+')
            f.write("\n VALID: ")
            f.write(str(valid))
            f.close()
            # early-stop with patience
            mrr_valid = valid['MRR']
            if mrr_valid < mrr_std:
               patience += 1
               if patience >= 100 and epoch>epoch_pretrain:
                  print("Early stopping ...")
                  break
            else:
               patience = 0
               mrr_std = mrr_valid
               torch.save(model.state_dict(), os.path.join(PATH, modelname+'.pkl'))

            curve['valid'].append(valid)
            # curve['test'].append(test)
            if not dataset.interval:
                curve['train'].append(train)
    
                print("\t TRAIN: ", train)
            print("\t VALID : ", valid)
            print("\t TEST : ", test)

    model.load_state_dict(torch.load(os.path.join(PATH, modelname+'.pkl')))
    results = avg_both(*dataset.eval(model, 'test', -1))
    print("\n\nTEST : ", results)
    f = open(os.path.join(PATH, 'result.txt'), 'w+')
    f.write("\n\nTEST : ")
    f.write(str(results))
    f.close()

if __name__ == '__main__':

    learn()


