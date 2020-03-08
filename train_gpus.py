
import os
import time
import shutil

import numpy

import data
from vocab import Vocabulary, deserialize_vocab
from model import SCAN, ContrastiveLoss
from evaluation import evalrank

import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_

import argparse
import opts

def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()

def main():
    # Hyper Parameters
    
    opt = opts.parse_opt()

    device_id = opt.gpuid
    device_count = len(str(device_id).split(","))
    #assert device_count == 1 or device_count == 2
    print("use GPU:", device_id, "GPUs_count", device_count, flush=True)
    os.environ['CUDA_VISIBLE_DEVICES']=str(device_id)
    device_id = 0
    torch.cuda.set_device(0)

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(
        opt.data_name, vocab, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = SCAN(opt)
    model.cuda()
    model = nn.DataParallel(model)

     # Loss and Optimizer
    criterion = ContrastiveLoss(opt=opt, margin=opt.margin, max_violation=opt.max_violation)
    mse_criterion = nn.MSELoss(reduction="batchmean")
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.learning_rate)

    # optionally resume from a checkpoint
    if not os.path.exists(opt.model_name):
        os.makedirs(opt.model_name)
    start_epoch = 0
    best_rsum = 0

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})"
                  .format(opt.resume, start_epoch, best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    evalrank(model.module, val_loader, opt)

    print(opt, flush=True)
    
    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        message = "epoch: %d, model name: %s\n" % (epoch, opt.model_name)
        log_file = os.path.join(opt.logger_name, "performance.log")
        logging_func(log_file, message)
        print("model name: ", opt.model_name, flush=True)
        adjust_learning_rate(opt, optimizer, epoch)
        run_time = 0
        for i, (images, captions, lengths, masks, ids, _) in enumerate(train_loader):
            start_time = time.time()
            model.train()

            optimizer.zero_grad()

            if device_count != 1:
                images = images.repeat(device_count,1,1)

            score = model(images, captions, lengths, masks, ids)
            loss = criterion(score)

            loss.backward()
            if opt.grad_clip > 0:
                clip_grad_norm_(model.parameters(), opt.grad_clip)
            optimizer.step()
            run_time += time.time() - start_time
            # validate at every val_step
            if i % 100 == 0:
                log = "epoch: %d; batch: %d/%d; loss: %.4f; time: %.4f" % (epoch, 
                            i, len(train_loader), loss.data.item(), run_time / 100)
                print(log, flush=True)
                run_time = 0
            if (i + 1) % opt.val_step == 0:
                evalrank(model.module, val_loader, opt)

        print("-------- performance at epoch: %d --------" % (epoch))
        # evaluate on validation set
        rsum = evalrank(model.module, val_loader, opt)
        #rsum = -100
        filename = 'model_' + str(epoch) + '.pth.tar'
        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
        }, is_best, filename=filename, prefix=opt.model_name + '/')


def save_checkpoint(state, is_best, filename='model.pth.tar', prefix=''):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                message = "--------save best model at epoch %d---------\n" % (state["epoch"]-1)
                print(message, flush=True)
                log_file = os.path.join(prefix, "performance.log")
                logging_func(log_file, message)
                shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print('model save {} failed, remaining {} trials'.format(filename, tries), flush=True)
        if not tries:
            raise error

def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    print("learning rate %f in epoch %d" % (lr, epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()
