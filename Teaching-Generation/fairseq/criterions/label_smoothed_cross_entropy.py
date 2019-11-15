# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math

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

    def forward(self, model, sample, teacher_model, eval, reduce=True):
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
            teacher_output = teacher_model(pivot_tokens, src_lengths, prev_output_tokens)
            teacher_prob = teacher_model.get_normalized_probs(teacher_output, log_probs=False)
            teacher_prob = teacher_prob.view(-1, teacher_prob.size(-1))

            loss = self.kl_div(lprobs, teacher_prob)
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
