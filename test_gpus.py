
import evaluation as evaluation
from vocab import Vocabulary, deserialize_vocab
from model import SCAN
import data

import argparse
import os
import torch
import torch.nn as nn

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets')
    parser.add_argument('--model_path', default='./data/',
                        help='path to model')
    parser.add_argument('--split', default='test',
                        help='val/test')
    parser.add_argument('--gpuid', default=0., type=str,
                        help='gpuid')
    parser.add_argument('--fold5', action='store_true',
                        help='fold5')
    opts = parser.parse_args()

    device_id = opts.gpuid
    print("use GPU:", device_id)
    os.environ['CUDA_VISIBLE_DEVICES']=str(device_id)
    device_id = 0
    torch.cuda.set_device(0)
    # load model and options
    checkpoint = torch.load(opts.model_path)
    opt = checkpoint['opt']
    opt.loss_verbose = False
    opt.split = opts.split
    opt.data_path = opts.data_path
    opt.fold5 = opts.fold5

    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = SCAN(opt)
    model.cuda()
    model = nn.DataParallel(model)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = data.get_test_loader(opt.split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print(opt)
    print('Computing results...')

    evaluation.evalrank(model.module, data_loader, opt, split=opt.split, fold5=opt.fold5)

if __name__ == '__main__':
    main()
