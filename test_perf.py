import argparse
import torch 
from torch import nn
from _asm import AdaptiveLogSoftmaxWithLoss
from time import perf_counter


def main(config):
    
    decoder = AdaptiveLogSoftmaxWithLoss(in_features=config.nhid, n_classes=config.ntoken, cutoffs=config.cutoff)

    input  = torch.FloatTensor(config.batch * config.niter, config.nhid).uniform_(-0.1, 0.1)
    target = torch.randint(low=0, high=config.ntoken, size=(config.batch * config.niter,), dtype=torch.long)

    decoder = decoder.to('cuda')
    input   = input.to('cuda')
    target  = target.to('cuda')

    idx = 0
    start = perf_counter()

    while idx < target.size(0):

        x = input.narrow(0, idx, config.batch)
        y = target.narrow(0, idx, config.batch)

        decoder.zero_grad()

        loss = decoder(x, y).loss
        loss.backward()

        idx = idx + config.batch
    
    stop = perf_counter()

    print(stop - start)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-nhid',    default=256, type=int)
    parser.add_argument('-batch',   default=64, type=int)
    parser.add_argument('-niter',   default=1000, type=int)
    parser.add_argument('-cutoff',  default='2000,10000')
    parser.add_argument('-ntoken',  default=100000, type=int)

    args = parser.parse_args()
    args.cutoff = [int(item) for item in args.cutoff.split(',')]

    main(args)
