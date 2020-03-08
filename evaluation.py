
from __future__ import print_function
import os

import sys
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import SCAN
from collections import OrderedDict
import time

def logging_func(log_file, message):
    with open(log_file,'a') as f:
        f.write(message)
    f.close()

def evalrank(model, data_loader, opt, split='dev', fold5=False):
    print("-------- evaluation --------")
    model.eval()
    with torch.no_grad():
        img_fc, img_embs, cap_ht, cap_embs, cap_lens = encode_data(model, data_loader)
        print('Images: %d, Captions: %d' %
              (img_embs.shape[0] / 5, cap_embs.shape[0]))

        if not fold5:
            # no cross-validation, full evaluation
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
            sims = shard_xattn(model, img_fc, img_embs, cap_ht, cap_embs, cap_lens, opt, shard_size=128)
            r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
            ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f" % rsum)
            print("Average i2t Recall: %.1f" % ar)
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
            print("Average t2i Recall: %.1f" % ari)
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
            
            message = "split: %s, Image to text: (%.1f, %.1f, %.1f) " % (split, r[0], r[1], r[2])
            message += "Text to image: (%.1f, %.1f, %.1f) " % (ri[0], ri[1], ri[2])
            message += "rsum: %.1f\n" % rsum

            log_file = os.path.join(opt.logger_name, "performance.log")
            logging_func(log_file, message)
            if split == "test" or split == "testall":
                #torch.save({'rt': rt, 'rti': rti}, os.path.join(opt.logger_name, 'ranks.pth.tar'))

                #torch.save({"sims_ti": sims_0, "sims_it": sims_1}, os.path.join(opt.logger_name, 'sims_seperate.pth.tar'))
                torch.save(sims, os.path.join(opt.logger_name, 'sims.pth.tar'))
        else:
            # 5fold cross-validation, only for MSCOCO
            results = []
            for i in range(5):
                img_fc_shard = img_fc[i * 5000:(i + 1) * 5000:5]
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                cap_ht_shard = cap_ht[i * 5000:(i + 1) * 5000]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
                sims = shard_xattn(model, img_fc_shard, img_embs_shard, cap_ht_shard, cap_embs_shard, cap_lens_shard, opt, shard_size=128)
                r, rt0 = i2t(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
                print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
                ri, rti0 = t2i(img_embs_shard, cap_embs_shard, cap_lens_shard, sims, return_ranks=True)
                print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

                if i == 0:
                    rt, rti = rt0, rti0
                ar = (r[0] + r[1] + r[2]) / 3
                ari = (ri[0] + ri[1] + ri[2]) / 3
                rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
                print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
                results += [list(r) + list(ri) + [ar, ari, rsum]]

            print("-----------------------------------")
            print("Mean metrics: ")
            mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
            print("rsum: %.1f" % (mean_metrics[10] * 6))
            print("Average i2t Recall: %.1f" % mean_metrics[11])
            print("Image to text: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[:5])
            print("Average t2i Recall: %.1f" % mean_metrics[12])
            print("Text to image: %.1f %.1f %.1f %.1f %.1f" %
                  mean_metrics[5:10])
            message = "split: %s, Image to text: (%.1f, %.1f, %.1f) " % (split, mean_metrics[0], mean_metrics[1], mean_metrics[2])
            message += "Text to image: (%.1f, %.1f, %.1f) " % (mean_metrics[5], mean_metrics[6], mean_metrics[7])
            message += "rsum: %.1f\n" % (mean_metrics[10] * 6)

            log_file = os.path.join(opt.logger_name, "performance.log")
            logging_func(log_file, message)
        return rsum

def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """

    # np array to keep all the embeddings
    img_fcs = None
    img_embs = None
    cap_hts = None
    cap_embs = None
    cap_lens = None

    max_n_word = 0
    for i, (images, captions, lengths, masks, ids, _) in enumerate(data_loader):
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, masks, ids, _) in enumerate(data_loader):

        # compute the embeddings
        img_fc, img_emb, cap_ht, cap_emb, cap_len = model.forward_emb(images, captions, lengths, masks)
        #print(img_emb)
        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            img_fcs = np.zeros((len(data_loader.dataset), img_fc.size(1)))

            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_hts = np.zeros((len(data_loader.dataset), cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        img_fcs[ids] = img_fc.data.cpu().numpy().copy()

        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
        cap_hts[ids] = cap_ht.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]
        del images, captions
    return img_fcs, img_embs, cap_hts, cap_embs, cap_lens

def shard_xattn(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128):
    
    if opt.model_mode == "full_IMRAM":
        sims = shard_xattn_Full_IMRAM(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128)
    elif opt.model_mode == "image_IMRAM":
        sims = shard_xattn_Image_IMRAM(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128)
    elif opt.model_mode == "text_IMRAM":
        sims = shard_xattn_Text_IMRAM(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128)
    else:
      assert False, "wrong model mode"
    return sims

def shard_xattn_Full_IMRAM(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size) + 1
    n_cap_shard = int((len(captions)-1)/shard_size) + 1
    
    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d_t2i = [np.zeros((len(images), len(captions))) for _ in range(opt.iteration_step)]
    d_i2t = [np.zeros((len(images), len(captions))) for _ in range(opt.iteration_step)]

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im_fc = torch.from_numpy(images_fc[im_start:im_end]).cuda()
            im_emb = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim_list_t2i = model.xattn_score_Text_IMRAM(im_fc, im_emb, h, s, l, opt)
            sim_list_i2t = model.xattn_score_Image_IMRAM(im_fc, im_emb, h, s, l, opt)
            assert len(sim_list_t2i) == opt.iteration_step and len(sim_list_i2t) == opt.iteration_step
            for k in range(opt.iteration_step):
                d_t2i[k][im_start:im_end, cap_start:cap_end] = sim_list_t2i[k].data.cpu().numpy()
                d_i2t[k][im_start:im_end, cap_start:cap_end] = sim_list_i2t[k].data.cpu().numpy()

    score = 0
    for j in range(opt.iteration_step):
        score += d_t2i[j]
    for j in range(opt.iteration_step):
        score += d_i2t[j]

    return score

def shard_xattn_Text_IMRAM(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size) + 1
    n_cap_shard = int((len(captions)-1)/shard_size) + 1
    
    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = [np.zeros((len(images), len(captions))) for _ in range(opt.iteration_step)]

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im_fc = torch.from_numpy(images_fc[im_start:im_end]).cuda()
            im_emb = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim_list = model.xattn_score_Text_IMRAM(im_fc, im_emb, h, s, l, opt)
            assert len(sim_list) == opt.iteration_step
            for k in range(opt.iteration_step):
                d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].data.cpu().numpy()

    score = 0
    for j in range(opt.iteration_step):
        score += d[j]

    return score

def shard_xattn_Image_IMRAM(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size) + 1
    n_cap_shard = int((len(captions)-1)/shard_size) + 1
    
    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = [np.zeros((len(images), len(captions))) for _ in range(opt.iteration_step)]

    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im_fc = torch.from_numpy(images_fc[im_start:im_end]).cuda()
            im_emb = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim_list = model.xattn_score_Image_IMRAM(im_fc, im_emb, h, s, l, opt)
            assert len(sim_list) == opt.iteration_step
            for k in range(opt.iteration_step):
                if len(sim_list[k]) != 0:
                    d[k][im_start:im_end, cap_start:cap_end] = sim_list[k].data.cpu().numpy()

    score = 0
    for j in range(opt.iteration_step):
        score += d[j]
    return score

def shard_xattn_t2i(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size) + 1
    n_cap_shard = int((len(captions)-1)/shard_size) + 1
    
    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))

    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim = model.xattn_score_t2i(im, h, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d

def shard_xattn_i2t(model, images_fc, images, caption_ht, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = int((len(images)-1)/shard_size) + 1
    n_cap_shard = int((len(captions)-1)/shard_size) + 1

    print("n_im_shard: %d, n_cap_shard: %d" % (n_im_shard, n_cap_shard))
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = torch.from_numpy(images[im_start:im_end]).cuda()
            h = torch.from_numpy(caption_ht[cap_start:cap_end]).cuda()
            s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            l = caplens[cap_start:cap_end]
            sim = model.xattn_score_i2t(im, h, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    return d

def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = sims.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
