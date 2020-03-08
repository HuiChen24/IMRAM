import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils.weight_norm import weight_norm
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F


def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def EncoderImage(data_name, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an different encoders
    that uses precomputed image features.
    """
    if precomp_enc_type == 'basic':
        img_enc = EncoderImagePrecomp(
            img_dim, embed_size, no_imgnorm)
    elif precomp_enc_type == 'weight_norm':
        img_enc = EncoderImageWeightNormPrecomp(
            img_dim, embed_size, no_imgnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))

    return img_enc

class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc_local = nn.Linear(img_dim, embed_size)
        #self.fc_global = nn.Linear(img_dim, embed_size)

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc_local.in_features +
                                  self.fc_local.out_features)
        self.fc_local.weight.data.uniform_(-r, r)
        self.fc_local.bias.data.fill_(0)

        #self.fc_global.weight.data.uniform_(-r, r)
        #self.fc_global.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        #img_global = images.mean(1)
        #feat_global = self.fc_global(img_global)
        feat_local = self.fc_local(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            feat_local = l2norm(feat_local, dim=-1)
            #feat_global = l2norm(feat_global, dim=-1)

        return feat_local.mean(1), feat_local 

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)

class EncoderImageWeightNormPrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, no_imgnorm=False):
        super(EncoderImageWeightNormPrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.fc_local = weight_norm(nn.Linear(img_dim, embed_size), dim=None)
        #self.fc_global = weight_norm(nn.Linear(img_dim, embed_size), dim=None)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized

        #img_global = images.mean(1)
        #feat_global = self.fc_global(img_global)
        feat_local = self.fc_local(images)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            feat_local = l2norm(feat_local, dim=-1)
            #feat_global = l2norm(feat_global, dim=-1)

        return feat_local.mean(1), feat_local 

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImageWeightNormPrecomp, self).load_state_dict(new_state)

# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, opt, vocab_size, word_dim, embed_size, num_layers,
                 use_bi_gru=False, no_txtnorm=False, pos_emb=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        # caption embedding
        self.use_bi_gru = use_bi_gru
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=use_bi_gru)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        
        sorted_lengths, indices = torch.sort(lengths, descending=True)
        x_emb = x_emb[indices]
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)

        packed = pack_padded_sequence(x_emb, sorted_lengths.data.tolist(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()
        # Forward propagate RNN
        out, ht = self.rnn(packed)

        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = cap_emb[inv_ix]
        cap_len = cap_len[inv_ix]
        ht[0] = ht[0][inv_ix]
        ht[1] = ht[1][inv_ix]

        if self.use_bi_gru:
            cap_emb = (cap_emb[:,:,:int(cap_emb.size(2)/2)] + cap_emb[:,:,int(cap_emb.size(2)/2):])/2
            ht = (ht[0] + ht[1]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)
            ht = l2norm(ht, dim=-1)

        # For multi-GPUs
        if cap_emb.size(1) < x_emb.size(1):
            pad_size = x_emb.size(1) - cap_emb.size(1)
            pad_emb = torch.Tensor(cap_emb.size(0), pad_size, cap_emb.size(2))
            if torch.cuda.is_available():
                pad_emb = pad_emb.cuda()
            cap_emb = torch.cat([cap_emb, pad_emb], 1)

        return ht, cap_emb, cap_len

def func_attention(query, context, opt, smooth, eps=1e-8, weight=None):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)

    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = F.softmax(attn, dim=1)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    
    if weight is not None:
      attn = attn + weight

    attn_out = attn.clone()

    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)

    attn = F.softmax(attn*smooth, dim=1)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attn_out

def cosine_similarity_a2a(x1, x2, dim=1, eps=1e-8):
    #x1: (B, n, d) x2: (B, m, d)
    w12 = torch.bmm(x1, x2.transpose(1,2))
    #w12: (B, n, m)

    w1 = torch.norm(x1, 2, dim).unsqueeze(2)
    w2 = torch.norm(x2, 2, dim).unsqueeze(1)

    #w1: (B, n, 1) w2: (B, 1, m)
    w12_norm = torch.bmm(w1, w2).clamp(min=eps)
    return w12 / w12_norm

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps))

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, scores):

        diagonal = scores.diag().view(-1, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class SCAN(nn.Module):
    """
    Stacked Cross Attention Network (SCAN) model
    """
    def __init__(self, opt):
        super(SCAN, self).__init__()
        # Build Models
        self.opt = opt
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    precomp_enc_type=opt.precomp_enc_type,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt, opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers, 
                                   use_bi_gru=opt.bi_gru,  
                                   no_txtnorm=opt.no_txtnorm)
        
        print("*********using gate to fusion information**************")
        self.linear_t2i = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.gate_t2i = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.linear_i2t = nn.Linear(opt.embed_size * 2, opt.embed_size)
        self.gate_i2t = nn.Linear(opt.embed_size * 2, opt.embed_size)

    def gated_memory_t2i(self, input_0, input_1):

        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = F.tanh(self.linear_t2i(input_cat))
        gate = torch.sigmoid(self.gate_t2i(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output
    
    def gated_memory_i2t(self, input_0, input_1):

        input_cat = torch.cat([input_0, input_1], 2)
        input_1 = F.tanh(self.linear_i2t(input_cat))
        gate = torch.sigmoid(self.gate_i2t(input_cat))
        output = input_0 * gate + input_1 * (1 - gate)

        return output

    def forward_emb(self, images, captions, lengths, masks):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            lengths = lengths.cuda()

        # Forward
        img_fc, img_emb = self.img_enc(images)

        ht, cap_emb, cap_lens = self.txt_enc(captions, lengths)
        return img_fc, img_emb, ht, cap_emb, lengths

    def forward_score(self, img_fc, img_emb, ht, cap_emb, cap_len, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        # compute image-sentence score matrix
        if self.opt.model_mode == "full_IMRAM":
            scores_t2i = self.xattn_score_Text_IMRAM(img_fc, img_emb, ht, cap_emb, cap_len, self.opt)
            scores_i2t = self.xattn_score_Image_IMRAM(img_fc, img_emb, ht, cap_emb, cap_len, self.opt)
            scores_t2i = torch.stack(scores_t2i, 0).sum(0)
            scores_i2t = torch.stack(scores_i2t, 0).sum(0)
            score = scores_t2i + scores_i2t
        elif self.opt.model_mode == "image_IMRAM":
            scores_i2t = self.xattn_score_Image_IMRAM(img_fc, img_emb, ht, cap_emb, cap_len, self.opt)
            scores_i2t = torch.stack(scores_i2t, 0).sum(0)
            score = scores_i2t
        elif self.opt.model_mode == "text_IMRAM":
            scores_t2i = self.xattn_score_Text_IMRAM(img_fc, img_emb, ht, cap_emb, cap_len, self.opt)
            scores_t2i = torch.stack(scores_t2i, 0).sum(0)
            score = scores_t2i
        return score

    def forward(self, images, captions, lengths, masks, ids=None, *args):
        """One training step given images and captions.
        """
        # compute the embeddings
        img_fc, img_emb, ht, cap_emb, cap_lens = self.forward_emb(images, captions, lengths, masks)
        scores = self.forward_score(img_fc, img_emb, ht, cap_emb, cap_lens)
        return scores
    
    def xattn_score_Text_IMRAM(self, images_fc, images, caption_ht, captions_all, cap_lens, opt):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = [[] for _ in range(opt.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            
            query = cap_i_expand
            context = images
            weight = 0
            for j in range(opt.iteration_step):
                # "feature_update" by default:
                attn_feat, _ = func_attention(query, context, opt, smooth=opt.lambda_softmax)

                row_sim = cosine_similarity(cap_i_expand, attn_feat, dim=2)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)

                query = self.gated_memory_t2i(query, attn_feat)

                if not opt.no_IMRAM_norm:
                    query = l2norm(query, dim=-1)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(opt.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0,1)
            new_similarities.append(similarities_one)
        
        return new_similarities

    def xattn_score_Image_IMRAM(self, images_fc, images, caption_ht, captions_all, cap_lens, opt):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = [[] for _ in range(opt.iteration_step)]
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        n_region = images.size(1)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            cap_h_i = caption_ht[i].unsqueeze(0).unsqueeze(0).contiguous()
            cap_h_i_expand = cap_h_i.expand_as(images)
            
            query = images
            context = cap_i_expand
            weight = 0
            for j in range(opt.iteration_step):
                attn_feat, _ = func_attention(query, context, opt, smooth=opt.lambda_softmax)

                row_sim = cosine_similarity(images, attn_feat, dim=2)
                row_sim = row_sim.mean(dim=1, keepdim=True)
                similarities[j].append(row_sim)

                query = self.gated_memory_i2t(query, attn_feat)

                if not opt.no_IMRAM_norm:
                    query = l2norm(query, dim=-1)

        # (n_image, n_caption)
        new_similarities = []
        for j in range(opt.iteration_step):
            if len(similarities[j]) == 0:
                new_similarities.append([])
                continue
            similarities_one = torch.cat(similarities[j], 1).double()
            if self.training:
                similarities_one = similarities_one.transpose(0,1)
            new_similarities.append(similarities_one)

        return new_similarities

    def xattn_score_t2i(self, images_fc, images, caption_ht, captions_all, cap_lens, opt):
        """
        Images: (n_image, n_regions, d) matrix of images
        captions_all: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        similarities = []
        weiContext_a2a = []
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d)
                attn: (n_image, n_region, n_word)
            """
            weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()

            weiContext_a2a.append(cap_i_expand.double() + weiContext.double())
                
            row_sim = cosine_similarity(cap_i_expand.double(), weiContext.double(), dim=2)

            if opt.agg_func == 'LogSumExp':
                row_sim.mul_(opt.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim)/opt.lambda_lse
            elif opt.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif opt.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif opt.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1).double()
        if self.training:
            similarities = similarities.transpose(0,1)
        
        return similarities, weiContext_a2a

    def xattn_score_i2t(self, images_fc, images, caption_ht, captions_all, cap_lens, opt):
        """
        Images: (batch_size, n_regions, d) matrix of images
        captions_all: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = []
        weiContext_a2a = []
        n_image = images.size(0)
        n_caption = captions_all.size(0)
        n_region = images.size(1)
        images = images.float()
        captions_all = captions_all.float()
        caption_ht = caption_ht.float()
        images_fc = images.mean(1, keepdim=True)
        for i in range(n_caption):
            # Get the i-th text description
            n_word = cap_lens[i]
            cap_i = captions_all[i, :n_word, :].unsqueeze(0).contiguous()
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            cap_h_i = caption_ht[i].unsqueeze(0).unsqueeze(0).contiguous()
            cap_h_i_expand = cap_h_i.expand_as(images)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_region, d)
                weiContext: (n_image, n_region, d)
                attn: (n_image, n_word, n_region)
            """
            weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
            
            weiContext_a2a.append(weiContext.double() + images.double())
            
            # (n_image, n_region)
            row_sim = cosine_similarity(images, weiContext, dim=2)
            if opt.agg_func == 'LogSumExp':
                row_sim.mul_(opt.lambda_lse).exp_()
                row_sim = row_sim.sum(dim=1, keepdim=True)
                row_sim = torch.log(row_sim)/opt.lambda_lse
            elif opt.agg_func == 'Max':
                row_sim = row_sim.max(dim=1, keepdim=True)[0]
            elif opt.agg_func == 'Sum':
                row_sim = row_sim.sum(dim=1, keepdim=True)
            elif opt.agg_func == 'Mean':
                row_sim = row_sim.mean(dim=1, keepdim=True)
            else:
                raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1).double()
        if self.training:
            similarities = similarities.transpose(0,1)
        return similarities, weiContext_a2a
