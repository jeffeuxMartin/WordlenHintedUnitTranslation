import math
from dataclasses import dataclass, field

import torch.nn as nn
from fairseq import metrics, utils
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)
from fairseq.criterions import FairseqCriterion, register_criterion

def compute_length_loss(output, target):
    length_crit = nn.L1Loss()
    return length_crit(output, target)

@dataclass
class AugLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    length_penalty: float = field(
        default=0.0,
        metadata={"help": "length_penalty"},
    )

@register_criterion(
    "aug_label_smoothed_cross_entropy", dataclass=AugLabelSmoothedCrossEntropyCriterionConfig
)
class AugLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, ignore_prefix_size=0, report_accuracy=False, length_penalty=0.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.length_penalty = length_penalty
        
    def forward(self, model, sample, reduce=True):
        net_output = model(**sample["net_input"])   # <--
        # length_loss = compute_length_loss(*length_diff)  # output, target   # <--
        if "length_diff" in net_output[1]:  # dict
            aug_loss, loss, nll_loss, length_loss = self.compute_loss(model, net_output, sample, reduce=reduce)  # <--
        else:
            loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)  # <--
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if "length_diff" in net_output[1]:  # dict
            logging_output["aug_loss"] = aug_loss.data  # <--
            logging_output["length_loss"] = length_loss.data  # <--
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return (
            aug_loss if "length_diff" in net_output[1] else loss,
            # loss + 0.1 * length_loss,  # <-- 
            sample_size, logging_output)
            
    def compute_loss(self, model, net_output, sample, reduce=True):
        loss, nll_loss = super().compute_loss(model, net_output, sample, reduce)
        if "length_diff" in net_output[1]:  # dict
            logits, others = net_output
            [length_diff] = others["length_diff"]
            # length_loss = compute_length_loss(*length_diff)  # output, target   # <--
            length_loss = (length_diff[0] - length_diff[1]).abs().sum()
            aug_loss = loss + self.length_penalty * length_loss
            return aug_loss, loss, nll_loss, length_loss
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        aug_loss_sum = sum(log.get("aug_loss", 0) for log in logging_outputs)
        length_loss_sum = sum(log.get("length_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "aug_loss", aug_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "length_loss", length_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True

        # breakpoint()
