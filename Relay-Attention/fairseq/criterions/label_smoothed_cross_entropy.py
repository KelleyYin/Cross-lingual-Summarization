# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
from fairseq import utils
import torch.nn.functional as F
from . import FairseqCriterion, register_criterion
import torch.nn as nn

@register_criterion('label_smoothed_cross_entropy')
class LabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
        self.kl_div = nn.KLDivLoss(size_average=False)

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')

    def relay_attn(self, MT_attn, NHG_attn, top_k, p):
        batch, tgt_len, src_len = NHG_attn.size()
        NHG_attn = NHG_attn.view(-1, src_len)
        NHG_top = NHG_attn.topk(top_k, dim=-1)[-1]
        offsets = torch.arange(0, batch*src_len, step=src_len).unsqueeze(1).\
            repeat(1, tgt_len).view(-1, 1).type_as(MT_attn).long()
        NHG_top = NHG_top + offsets
        MT_attn_ = MT_attn.contiguous().view(-1, MT_attn.size(-1))
        MT_max = MT_attn_.max(dim=-1, keepdim=True)[-1]
        tmp = torch.take(MT_max, NHG_top)

        teach_attn = torch.zeros(batch*tgt_len, MT_attn.size(-1)).type_as(MT_attn)
        ara = torch.arange(teach_attn.size(0)).unsqueeze(1).type_as(tmp)

        teach_attn[ara, tmp] = float(p)
        teach_attn = teach_attn.view(batch, tgt_len, -1)

        return teach_attn

    def forward(self, model, sample, NHG_teach, MT_teach, eval, update_attn, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        src_tokens = sample['net_input']["src_tokens"]
        src_lengths = sample['net_input']['src_lengths']
        prev_output_tokens = sample['net_input']['prev_output_tokens']

        net_output = model(src_tokens, src_lengths, prev_output_tokens)
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        target = model.get_targets(sample, net_output).view(-1, 1)
        non_pad_mask = target.ne(self.padding_idx)

        if not eval:
            pivot_tokens = sample['net_input']['pivot_tokens']
            NHG_output = NHG_teach(pivot_tokens, src_lengths, prev_output_tokens)
            NHG_probs = NHG_teach.get_normalized_probs(NHG_output, log_probs=False)
            NHG_probs = NHG_probs.view(-1, NHG_probs.size(-1))

            if update_attn:
                mt_src_tokens = sample['net_input']['mt_src_tokens']
                mt_tgt_tokens = sample['net_input']['mt_tgt_tokens']

                NHG_attn = NHG_output[-1]
                MT_attn = MT_teach(mt_src_tokens, src_lengths, mt_tgt_tokens)
                MT_attn = MT_attn.transpose(1, 2)

                teach_attn = self.relay_attn(MT_attn, NHG_attn, top_k=3, p=0.3)

                src_pad_mask = src_tokens.eq(self.padding_idx)
                if src_pad_mask.any():
                    teach_attn.masked_fill_(src_pad_mask.unsqueeze(1), 0.0)

                stu_attn = net_output[-1]
                attn_loss = F.mse_loss(stu_attn, teach_attn, size_average=False).sqrt()
                probs_kl_loss = self.kl_div(lprobs, NHG_probs)
                loss = attn_loss + probs_kl_loss
                nll_loss = attn_loss
            else:
                loss = self.kl_div(lprobs, NHG_probs)
                nll_loss = loss
        else:
            nll_loss = -lprobs.gather(dim=-1, index=target)[non_pad_mask]
            smooth_loss = -lprobs.sum(dim=-1, keepdim=True)[non_pad_mask]

            if reduce:
                nll_loss = nll_loss.sum()
                smooth_loss = smooth_loss.sum()

            eps_i = self.eps / lprobs.size(-1)
            loss = (1. - self.eps) * nll_loss + eps_i * smooth_loss

        pred_t = lprobs.data.max(1)[1]
        num_correct = pred_t.unsqueeze(-1).eq(target.data).masked_select(non_pad_mask.data).sum()

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(nll_loss.data) if reduce else nll_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
            "num_correct": utils.item(num_correct) if reduce else num_correct
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        return {
            'loss': sum(log.get('loss', 0) for log in logging_outputs) / sample_size / math.log(2),
            'nll_loss': sum(log.get('nll_loss', 0) for log in logging_outputs) / ntokens / math.log(2),
            'sample_size': sample_size,
        }
