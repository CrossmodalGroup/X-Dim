"""VSE model"""
import numpy as np

import torch
import torch.nn as nn

import torch.nn.functional as F

import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils import clip_grad_norm_

from lib.encoders import get_image_encoder, get_text_encoder
from lib.loss import ContrastiveLoss

import logging

logger = logging.getLogger(__name__)


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class Sim_vec(nn.Module):

    def __init__(self, embed_dim, opt):
        super(Sim_vec, self).__init__()

        self.opt = opt
        self.sub_dim = embed_dim

        self.weights = torch.nn.Embedding(embed_dim, embed_dim)
        self.sim_eval = nn.Linear(embed_dim, 1, bias=False)
        self.temp_scale = nn.Linear(1, 1, bias=False)
        self.sim_map = nn.Linear(embed_dim, 1, bias=False)
        self.temp_learnable = nn.Linear(1, 1, bias=False)
        self.init_weight()

    def init_weight(self):
        self.temp_scale.weight.data.fill_(np.log(1 / 0.07))
        self.temp_learnable.weight.data.fill_(4)
        # self.sim_map.weight.data.fill_(0.4) 
        self.sim_map.weight = nn.init.normal_(self.sim_map.weight,  mean=2.5, std=0.1)


    def forward(self, img_emb, cap_emb, cap_lens):
        n_image = img_emb.size(0)
        n_caption = cap_emb.size(0)
        sim_all = []

        sub_dim_index = torch.as_tensor(torch.linspace(0, self.sub_dim, steps=self.sub_dim, dtype=torch.int)).long().cuda()
        weights = torch.sigmoid(self.weights(sub_dim_index))

        ## sparse joint probability modeling
        joint_probability = (weights) #torch.sigmoid
        mean_probability = torch.mean(joint_probability, 1)
        std_probability = torch.std(joint_probability, 1)

        thres_probabilty = mean_probability + self.sim_map.weight * std_probability
        values = torch.exp(self.temp_learnable.weight) * (joint_probability - thres_probabilty.repeat(1024, 1).transpose(0, 1))
        mask_probability = torch.tanh(torch.exp(values))
        Dim_learned_weights = mask_probability * weights
        Dim_learned_weights = l2norm(Dim_learned_weights, 1)


        # # diagonal for bilinear operation imposing on similarity matrix
        # # Note that applying the proposed Agg(·) directly on the similarity matrix will result in a huge memory burden. According to theoretical analysis Sec.3.3.3, we can simplify it.
        Diagonal  = Dim_learned_weights.sum(0)
        Diagonal_Mask = torch.diag_embed(Diagonal)


        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_emb[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            ##################################################################################################
            query = cap_i_expand
            smooth = torch.exp(self.temp_scale.weight)
            # --> (batch, d, queryL)
            # (batch, sourceL, d)(batch, d, queryL)
            attn = torch.tanh(query @ Diagonal_Mask @ torch.transpose(img_emb, 1, 2))

            # --> (batch, sourceL, queryL)
            attnT = torch.transpose(attn, 1, 2).contiguous()
            attn = nn.LeakyReLU(0.1)(attnT)
            attn = l2norm(attn, 2)

            # --> (batch, queryL, sourceL)
            attn = torch.transpose(attn, 1, 2).contiguous()
            # --> (batch, queryL, sourceL
            attn = F.softmax(attn * smooth, dim=2)
            # --> (batch, queryL, d)
            Context_img = l2norm(torch.bmm(attn, img_emb), dim=-1)
            ##################################################################################################
            sim_loc = torch.mul(Context_img, cap_i_expand)

            sim_sub_space = sim_loc @ Dim_learned_weights.t()
            sim_loc = l2norm((sim_sub_space), dim=-1)
            sim_vec = sim_loc.sum(-1)
            sim = sim_vec.mean(dim=1, keepdim = True)

            sim_all.append(sim)

        sim_all = torch.cat(sim_all, 1)

        return sim_all

class VSEModel(object):
    """
        The standard VSE model
    """

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.img_enc = get_image_encoder(opt.data_name, opt.img_dim, opt.embed_size,
                                         precomp_enc_type=opt.precomp_enc_type,
                                         backbone_source=opt.backbone_source,
                                         backbone_path=opt.backbone_path,
                                         no_imgnorm=opt.no_imgnorm)
        self.txt_enc = get_text_encoder(opt.embed_size, no_txtnorm=opt.no_txtnorm)

        self.sim_vec = Sim_vec(opt.embed_size, opt)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            self.sim_vec.cuda()
            cudnn.benchmark = True

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)

        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())
        params += list(self.sim_vec.parameters())

        self.params = params
        self.opt = opt

        # Set up the lr for different parts of the VSE model
        decay_factor = 1e-4
        if opt.precomp_enc_type == 'basic':
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.sim_vec.parameters(), 'lr': opt.learning_rate},
                ],
                    lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD(self.params, lr=opt.learning_rate, momentum=0.9)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))
        else:
            if self.opt.optim == 'adam':
                all_text_params = list(self.txt_enc.parameters())
                bert_params = list(self.txt_enc.bert.parameters())
                bert_params_ptr = [p.data_ptr() for p in bert_params]
                text_params_no_bert = list()
                for p in all_text_params:
                    if p.data_ptr() not in bert_params_ptr:
                        text_params_no_bert.append(p)
                self.optimizer = torch.optim.AdamW([
                    {'params': text_params_no_bert, 'lr': opt.learning_rate},
                    {'params': bert_params, 'lr': opt.learning_rate * 0.1},
                    {'params': self.img_enc.backbone.top.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.backbone.base.parameters(),
                     'lr': opt.learning_rate * opt.backbone_lr_factor, },
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, weight_decay=decay_factor)
            elif self.opt.optim == 'sgd':
                self.optimizer = torch.optim.SGD([
                    {'params': self.txt_enc.parameters(), 'lr': opt.learning_rate},
                    {'params': self.img_enc.backbone.parameters(), 'lr': opt.learning_rate * opt.backbone_lr_factor,
                     'weight_decay': decay_factor},
                    {'params': self.img_enc.image_encoder.parameters(), 'lr': opt.learning_rate},
                ], lr=opt.learning_rate, momentum=0.9, nesterov=True)
            else:
                raise ValueError('Invalid optim option {}'.format(self.opt.optim))

        logger.info('Use {} as the optimizer, with init lr {}'.format(self.opt.optim, opt.learning_rate))

        self.Eiters = 0
        self.data_parallel = False

    def set_max_violation(self, max_violation):
        if max_violation:
            self.criterion.max_violation_on()
        else:
            self.criterion.max_violation_off()

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict(),
                      self.sim_vec.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0], strict=False)
        self.txt_enc.load_state_dict(state_dict[1], strict=False)
        self.sim_vec.load_state_dict(state_dict[2], strict=False)

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()
        self.sim_vec.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()
        self.sim_vec.eval()

    def freeze_backbone(self):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.freeze_backbone()
            else:
                self.img_enc.freeze_backbone()

    def unfreeze_backbone(self, fixed_blocks):
        if 'backbone' in self.opt.precomp_enc_type:
            if isinstance(self.img_enc, nn.DataParallel):
                self.img_enc.module.unfreeze_backbone(fixed_blocks)
            else:
                self.img_enc.unfreeze_backbone(fixed_blocks)

    def make_data_parallel(self):
        self.img_enc = nn.DataParallel(self.img_enc)
        self.txt_enc = nn.DataParallel(self.txt_enc)
        self.sim_vec = nn.DataParallel(self.sim_vec)
        self.data_parallel = True
        logger.info('Image encoder is data paralleled now.')

    @property
    def is_data_parallel(self):
        return self.data_parallel

    def forward_emb(self, images, captions, lengths, image_lengths=None):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if self.opt.precomp_enc_type == 'basic':
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
                image_lengths = image_lengths.cuda()
            img_emb = self.img_enc(images, image_lengths)
        else:
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()
            img_emb = self.img_enc(images)

        # lengths = torch.Tensor(lengths).cuda()
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, lengths

    def forward_sim(self, img_emb, cap_emb, cap_lens):

        sim_all = self.sim_vec(img_emb, cap_emb, cap_lens)

        return sim_all

    def forward_loss(self, sims):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(sims)
        self.logger.update('Le', loss.data.item(), sims.size(0))
        return loss

    def train_emb(self, images, captions, lengths, image_lengths=None, warmup_alpha=None):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, cap_lens = self.forward_emb(images, captions, lengths, image_lengths=image_lengths)
        sims = self.forward_sim(img_emb, cap_emb, cap_lens)

        # measure accuracy and record loss
        self.optimizer.zero_grad()
        loss = self.forward_loss(sims)

        if warmup_alpha is not None:
            loss = loss * warmup_alpha

        # compute gradient and update
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm_(self.params, self.grad_clip)
        self.optimizer.step()