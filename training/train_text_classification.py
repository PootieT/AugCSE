import json
import os
import sys
import warnings
from dataclasses import dataclass
import random
from enum import Enum
from typing import List, Dict, Optional, Union, Tuple, Callable

import torch
import wandb
import numpy as np
from torch import nn
import torch.nn.functional as F
from datasets import load_dataset, Dataset, load_metric
from transformers import (
    Trainer,
    AutoTokenizer,
    AutoConfig,
    CONFIG_MAPPING,
    default_data_collator,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    EvalPrediction,
    is_torch_tpu_available,
    TrainerCallback,
    TrainingArguments,
    TrainerState,
    TrainerControl,
    BertForPreTraining,
)
from transformers.modeling_utils import unwrap_model
from transformers.file_utils import PaddingStrategy
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import HPSearchBackend, PREFIX_CHECKPOINT_DIR

from evaluation.evaluate_similarity import get_huggingface_dataset, glue_task_to_keys

# Grossly adapted from SimCSE https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py

# Set path to SentEval
from testing_part.src.experimenters.sts_experimenter import SentExperimenter
from testing_part.src.test_datasets.cls_dataset import CLSDataset
from testing_part.src.test_datasets.nli_dataset import NLIDataset
from training import generate_text_classification
from training.model import ContrastiveModel
from training.utils import initialize_implementations
from transformations.random_word_augmentation import RandomDeletion

if sys.path[0].split("/")[-1] == "experiments":
    PATH_TO_SENTEVAL = "../SentEval"
    PATH_TO_DATA = "../SentEval/data"
else:
    PATH_TO_SENTEVAL = "./SentEval"
    PATH_TO_DATA = "./SentEval/data"

# Import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


class DistanceMetric(Enum):
    """
    distance metrics for contrastive loss
    """

    EUCLIDEAN = lambda x, y: torch.cdist(x.contiguous(), y.contiguous(), p=2.0)
    MANHATTAN = lambda x, y: torch.cdist(x.contiguous(), y.contiguous(), p=1.0)
    COSINE = lambda x, y: (1 - nn.CosineSimilarity(dim=-1)(x.unsqueeze(1), y.unsqueeze(0)))


@dataclass
class OurDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    do_mlm: bool = False
    mlm_probability: float = 0.15

    def __call__(
        self,
        features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]],
    ) -> Dict[str, torch.Tensor]:
        special_keys = [k for k in ["input_ids", "attention_mask", "token_type_ids"] if k in features[0]]

        if isinstance(features[0]["input_ids"][0], int):  # during eval
            for feature in features:
                for k in special_keys:
                    feature[k] = [feature[k]]
        bs = len(features)
        if bs > 0:
            num_sent = len(features[0]["input_ids"])
        else:
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.do_mlm:
            special_keys.extend(["mlm_input_ids", "mlm_labels"])
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

        batch = {
            k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
            for k in batch
        }

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        return batch

    def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """
        inputs = inputs.clone()
        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


class TrainingDataTransformationCallback(TrainerCallback):
    def __init__(self):
        pass

    @staticmethod
    def update_seed_args(args):
        try:
            init_args = json.loads(args.data_args.augmentation_init_args)
        except:
            init_args = {}
        if "seed" in init_args:
            init_args["seed"] += 1
        else:
            init_args["seed"] = 1  # end of epoch 0
        args.data_args.augmentation_init_args = json.dumps(init_args)

    def on_epoch_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        # create new perturbation data and reload training_dataset
        if (
            (args.data_args.regenerate_augmentation_per_epoch or args.data_args.resample_augmentation_per_epoch)
            and len(args.data_args.augmentations) > 0
            and args.num_train_epochs > 1
        ):
            print("Re-augmenting or Re-sampling training data ...")

            # update seed args so augmentations are not the same between epochs
            self.update_seed_args(args)
            impls = initialize_implementations(args.data_args.augmentations, args.data_args.augmentation_init_args)
            if args.data_args.regenerate_augmentation_per_epoch:
                # regenerate training data with new augmentation runs
                data_path = generate_text_classification.generate(
                    impls, args.data_args.dataset_name, "train[:100%]", data_args=args.data_args
                )
                args.data_args.train_file = data_path

            # re-sample the dataset existing in the path
            datasets = get_dataset(args.data_args, args)
            if args.use_pos_neg_pipeline:
                train_dataset = prepare_train_feature_pos_neg_batch(
                    datasets, kwargs["tokenizer"], args.data_args, state.epoch
                )
            else:
                train_dataset = prepare_train_feature_one_aug_batch(
                    datasets, kwargs["tokenizer"], args.data_args, state.epoch
                )
            # TODO very janky setup, may cause issues downstream
            kwargs["train_dataloader"].__initialized = False
            kwargs["train_dataloader"]._DataLoader__initialized = False
            kwargs["train_dataloader"].dataset = train_dataset
            kwargs["train_dataloader"]._DataLoader__initialized = True
            kwargs["train_dataloader"].__initialized = False


class ContrastiveTrainer(Trainer):
    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        eval_original: bool = False,
        eval_senteval_transfer: bool = False,
        eval_robust: bool = False,
    ) -> Dict[str, float]:

        # SentEval prepare and batcher
        def prepare(params, samples):
            return

        def batcher(params, batch):
            if isinstance(batch[0], list):
                sentences = [" ".join(s) for s in batch]
            else:
                sentences = batch
            batch = self.tokenizer.batch_encode_plus(
                sentences,
                return_tensors="pt",
                padding=True,  # truncation=True as well?
            )
            for k in batch:
                batch[k] = batch[k].to(self.args.device)
            with torch.no_grad():
                outputs = self.model(**batch, output_hidden_states=True, return_dict=True, eval=True)
                pooler_output = outputs.hidden_states[-1].cpu()
            return pooler_output

        # Set params for SentEval (fastmode)
        params = {"task_path": PATH_TO_DATA, "usepytorch": True, "kfold": 5}
        params["classifier"] = {"nhid": 0, "optim": "rmsprop", "batch_size": 128, "tenacity": 3, "epoch_size": 2}

        se = senteval.engine.SE(params, batcher, prepare)
        tasks = ["STSBenchmark", "SICKRelatedness"]
        if eval_senteval_transfer or self.args.eval_transfer:
            tasks = ["STSBenchmark", "SICKRelatedness", "MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]
        self.model.eval()
        results = se.eval(tasks)

        stsb_spearman = results["STSBenchmark"]["dev"]["spearman"][0]
        sickr_spearman = results["SICKRelatedness"]["dev"]["spearman"][0]

        metrics = {
            "eval_stsb_spearman": stsb_spearman,
            "eval_sickr_spearman": sickr_spearman,
            "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2,
        }
        if eval_senteval_transfer or self.args.eval_transfer:
            avg_transfer = 0
            for task in ["MR", "CR", "SUBJ", "MPQA", "SST2", "TREC", "MRPC"]:
                avg_transfer += results[task]["devacc"]
                metrics["eval_{}".format(task)] = results[task]["devacc"]
            avg_transfer /= 7
            metrics["eval_avg_transfer"] = avg_transfer

        if eval_original or self.args.eval_original:
            original_eval_metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            metrics.update(original_eval_metrics)

        if eval_robust or self.args.eval_robust:
            sts_experimenter = SentExperimenter()
            robust_test_dir = "/".join(self.args.output_dir.split("/")[:-4] + ["robust_test"])
            os.makedirs(robust_test_dir, exist_ok=True)
            # evaluate COLA
            cls_train_dataset = CLSDataset(split_type="train", base_dir=robust_test_dir)
            cls_val_dataset = CLSDataset(base_dir=robust_test_dir)
            acc = sts_experimenter.cola_experiment_with_batcher(
                cls_train_dataset, cls_val_dataset, batcher_func=batcher, batch_size=128, eval_type="linear"
            )
            metrics["cola_accuracy"] = acc

            # evaluate sts-mnli accuracy

            nli_dataset = NLIDataset(base_dir=robust_test_dir)
            acc = sts_experimenter.mnli_experiment_with_batcher(nli_dataset, batcher, batch_size=128)
            metrics["sts_mnli_accuracy"] = acc
        self.log(metrics)
        return metrics

    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Compared to original implementation, we change the saving policy to
        only save the best-validation checkpoints.
        """

        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save.
        assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Determine the new best metric / best model checkpoint
        if self.state.best_metric is None:
            self.state.best_metric = {}
        if metrics is not None and self.args.metric_for_best_model is not None:
            metrics_to_check = self.args.metric_for_best_model.split(",")
            for metric_to_check in metrics_to_check:
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                metric_value = metrics[metric_to_check]

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_model_checkpoint is None
                    or metric_to_check not in self.state.best_metric
                    or operator(metric_value, self.state.best_metric.get(metric_to_check))
                ):
                    # TODO enable multiple metrics saving n checkpoints if necessary
                    # output_dir = self.args.output_dir
                    output_dir = f"{self.args.output_dir}/{metric_to_check}"
                    self.state.best_metric[metric_to_check] = metric_value
                    self.state.best_model_checkpoint = output_dir

                    # Only save model when it is the best one
                    self.save_model(output_dir)
                    # if self.deepspeed:
                    #     self.deepspeed.save_checkpoint(output_dir)
                    #
                    # # Save optimizer and scheduler
                    # if self.sharded_dpp:
                    #     self.optimizer.consolidate_state_dict()

                    if is_torch_tpu_available():
                        xm.rendezvous("saving_optimizer_states")
                        xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            reissue_pt_warnings(caught_warnings)
                    elif self.is_world_process_zero() and not self.deepspeed:
                        # deepspeed.save_checkpoint above saves model/optim/sched
                        torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        reissue_pt_warnings(caught_warnings)

                    # Save the Trainer state
                    if self.is_world_process_zero():
                        self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
        else:
            # Save model checkpoint
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is not None and trial is not None:
                if self.hp_search_backend == HPSearchBackend.OPTUNA:
                    run_id = trial.number
                else:
                    from ray import tune

                    run_id = tune.get_trial_id()
                run_name = self.hp_name(trial) if self.hp_name is not None else f"run-{run_id}"
                output_dir = os.path.join(self.args.output_dir, run_name, checkpoint_folder)
            else:
                output_dir = os.path.join(self.args.output_dir, checkpoint_folder)

                self.store_flos()

            self.save_model(output_dir)
            # if self.deepspeed:
            #     self.deepspeed.save_checkpoint(output_dir)
            #
            # # Save optimizer and scheduler
            # if self.sharded_dpp:
            #     self.optimizer.consolidate_state_dict()

            if is_torch_tpu_available():
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    reissue_pt_warnings(caught_warnings)
            elif self.is_world_process_zero() and not self.deepspeed:
                # deepspeed.save_checkpoint above saves model/optim/sched
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                reissue_pt_warnings(caught_warnings)

            # Save the Trainer state
            if self.is_world_process_zero():
                self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))

            # Maybe delete some older checkpoints.
            if self.is_world_process_zero():
                self._rotate_checkpoints(use_mtime=True)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if not hasattr(self, "custom_metrics"):
            self.custom_metrics = {"uniform_loss": [], "alignment_loss": []}
            for k in ["discriminator_loss", "discriminator_accuracy", "discriminator_no_aug_predict_percent"]:
                if hasattr(model, k):
                    self.custom_metrics[k] = []
        else:
            for k in self.custom_metrics.keys():
                self.custom_metrics[k].append(getattr(model, k))

        if self.control.should_log:
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            logs["contrastive_loss"] = round(model.contrastive_loss, 4)
            logs["supervised_orig_loss"] = round(model.supervised_orig_loss, 4)
            if hasattr(model, "supervised_aug_loss"):
                logs["supervised_aug_loss"] = round(model.supervised_aug_loss, 4)
            for k in self.custom_metrics.keys():
                logs[k] = round(
                    np.array(self.custom_metrics[k]).sum().item()
                    / (self.state.global_step - self._globalstep_last_logged),
                    4,
                )
                self.custom_metrics[k] = []  # clearing storage

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, epoch, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def multiple_negative_ranking_loss_no_in_batch_negatives(self, z1, z2, z3):
        """
        METHOD 1: MultipleNegativeRankingLoss with positive and negatives
        given anchor (z1), positives (z2), and negatives(z3)
        similar loss to SimCSE with hard negatives, except we remove the negative
        augmentations's in-batch comparisons
        """
        temp = self.model_args.temp
        sim = (1 - self.model_args.distance_metric(z1, z2)) / temp

        # Hard negative
        z1_z3_sim = (1 - self.model_args.distance_metric(z1, z3)) / temp
        z1_z3_sim = torch.diag(z1_z3_sim).unsqueeze(1)
        cos_sim = torch.cat([sim, z1_z3_sim], 1)  # batch x (batch + 1)

        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        # Note that weights are actually logits of weights
        cos_sim[:, -1] += self.model_args.hard_negative_weight  # last column is hard negatives

        loss = loss_fct(cos_sim, labels)

        return loss

    def multiple_negative_ranking_loss_with_positives(self, z1, z2, z3):
        """
        METHOD 1: MultipleNegativeRankingLoss with positive and negatives
        given anchor (z1), positives (z2), and negatives(z3)
        Similar to MultipleNegativeRankingLoss in SBERT.com,
        same loss used by SimCSE with hard negatives
        """
        temp = self.model_args.temp
        sim = (1 - self.model_args.distance_metric(z1, z2)) / temp

        # Hard negative
        z1_z3_sim = (1 - self.model_args.distance_metric(z1, z3)) / temp
        cos_sim = torch.cat([sim, z1_z3_sim], 1)

        labels = torch.arange(cos_sim.size(0)).long().to(z1.device)
        loss_fct = nn.CrossEntropyLoss()

        # Calculate loss with hard negatives
        # Note that weights are actually logits of weights
        z3_weight = self.model_args.hard_negative_weight
        weights = torch.tensor(
            [
                [0.0] * (cos_sim.size(-1) - z1_z3_sim.size(-1))
                + [0.0] * i
                + [z3_weight]
                + [0.0] * (z1_z3_sim.size(-1) - i - 1)
                for i in range(z1_z3_sim.size(-1))
            ]
        ).to(z1.device)
        cos_sim = cos_sim + weights

        loss = loss_fct(cos_sim, labels)

        cos_sim_t = torch.cat([sim, cos_sim[:, cos_sim.size(0) :]], 0)
        loss_t = loss_fct(cos_sim_t.T, labels.T)
        loss = (loss + loss_t) / 2

        return loss

    def multiple_negative_ranking_loss(self, z1, z2, contrastive_labels):
        """
        METHOD 1: MultipleNegativeRankingLoss Loss + Uniform loss
        can't use cross entropy loss unless we define arbitrary positives at batches with no positives.
        of course one way we could do is to do cross entropy for rows in batches that have positives, and
        just push other samples uniformly away.

        Similar to MultipleNegativeRankingLoss in SBERT.com, same loss used by SimCSE
        """
        bs = len(z1)
        temp = self.model_args.temp
        t = self.model_args.uniform_t
        dist = self.model_args.distance_metric(z1, z2)
        sim = (1 - dist) / temp

        loss_fct = nn.CrossEntropyLoss()
        ce_loss = loss_fct(
            sim[contrastive_labels.bool(), :],
            torch.masked_select(torch.arange(bs).long().to(z1.device), contrastive_labels.bool()),
        )
        ce_loss_2 = loss_fct(
            sim[contrastive_labels.bool(), :].T,
            torch.masked_select(torch.arange(bs).long().to(z1.device), contrastive_labels.bool()).T,
        )
        ce_loss = (ce_loss_2 + ce_loss) / 2
        # TODO may need to use l2 distance for original paper
        if sum(~contrastive_labels.bool()) > 0:
            neg_dist = dist[:, ~contrastive_labels.bool()]
            uniform_loss = neg_dist.view(-1).pow(2).mul(-t).exp().mean().log()
            # self.model.contrastive_loss = uniform_loss.detach().cpu().item()
            ce_loss += uniform_loss

        return ce_loss

    def triplet_loss(self, z1, z2, contrastive_labels):
        """
        METHOD 2: Triplet Loss
        For each data point, treat each in-batch as negatives, treat augmentation as positive / negative
        multiply augmentation by aug label, subtract from in-batch negative distance, add margin
        loss = F.relu(distance_pos - distance_neg + self.triple_margin)

        # TODO maybe also try ContrastiveLoss or OnlineContrastiveLoss https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/ContrastiveLoss.py
        """
        bs = len(z1)
        temp = self.model_args.temp
        margin = self.model_args.margin
        dist = self.model_args.distance_metric(z1, z2) / temp

        # multiplying diagonal with contrastive label, this way negative labels are not used to subtract
        diff_sim = (torch.diag(dist) * contrastive_labels).unsqueeze(0).repeat(bs, 1) - dist
        diff_pairs = torch.masked_select(diff_sim, ~torch.eye(bs).bool().to(z1.device))
        loss = F.relu(diff_pairs + margin).mean()
        return loss

    def alignment_and_uniformity_loss(
        self,
        z1,
        z2,
        contrastive_labels,
        distance_func: DistanceMetric = None,
        alpha: Optional[int] = None,
        t: Optional[int] = None,
        temp: Optional[float] = None,
    ):
        """
        METHOD 3: Alignment and Uniformity loss, original loss uses l2 norm distance, here we
        provide the option for cosine as well as p2 (original formulation)
        """
        if distance_func is None:
            distance_func = self.model_args.distance_metric

        bs = len(z1)
        t = self.model_args.uniform_t if t is None else t
        alpha = self.model_args.alignment_alpha if alpha is None else alpha
        temp = self.model_args.temp if temp is None else temp

        dist = distance_func(z1, z2) / temp
        align_loss = (torch.diag(dist) * contrastive_labels).pow(alpha).mean()

        neg_mask = torch.eye(bs).bool().to(z1.device)
        neg_mask[range(bs), range(bs)] = contrastive_labels.bool()
        uniform_loss = torch.masked_select(dist, ~neg_mask).pow(2).mul(-t).exp().mean().log()
        self.model.uniform_loss = uniform_loss.detach().cpu().item()
        self.model.alignment_loss = align_loss.detach().cpu().item()
        return align_loss + uniform_loss

    def contrastive_loss(self, z1, z2, contrastive_labels):
        """
        METHOD 3: Contrastive loss, same as ContrastiveLoss in SBERT.com. We maximize distance between
        negative pairs and minimize distance between positive pairs. Only difference here is this
        considers all batch samples as negatives in addition to provided negatives
        """
        bs = len(z1)
        temp = self.model_args.temp
        m = self.model_args.margin

        dist = self.model_args.distance_metric(z1, z2) / temp
        pos_loss = (torch.diag(dist) * contrastive_labels).pow(2).mean()
        neg_mask = torch.eye(bs).bool().to(z1.device)
        neg_mask[range(bs), range(bs)] = contrastive_labels.bool()
        neg_loss = F.relu(m - torch.masked_select(dist, ~neg_mask)).pow(2).mean()
        return pos_loss + neg_loss

    def hard_contrastive_loss(self, z1, z2, contrastive_labels):
        """
        METHOD 3: Contrastive loss with hard negative/positives, same as OnlineContrastiveLoss in SBERT.com.
        We maximize distance between hardest negative pairs (smallest distance) and minimize distance
        between hardest positive pairs (farthest distance). Only difference here is this iconsiders all batch samples as negatives in addition to provided negatives
        """
        bs = len(z1)
        temp = self.model_args.temp
        m = self.model_args.margin

        dist = self.model_args.distance_metric(z1, z2) / temp

        poss = torch.diag(dist)[contrastive_labels.bool()]
        neg_mask = torch.eye(bs).bool().to(z1.device)
        neg_mask[range(bs), range(bs)] = contrastive_labels.bool()
        negs = torch.masked_select(dist, ~neg_mask)

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(m - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        return loss

    def eval_supervised_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs, eval=True)
        return (outputs.loss, outputs) if return_outputs else outputs.loss

    def compute_loss_one_aug(self, model, inputs, return_outputs=False):
        unsupervised_training = True if "labels" not in inputs else False
        if self.model_args.contrastive_loss_type in ["ranking_pos_neg", "ranking_neg"] or self.model_args.diff_cse_mode:
            # if using positive negative contrasting, add origin sentence again for second pass as positive
            for k in ["input_ids", "attention_mask", "token_type_ids"]:
                if k in inputs:
                    # TODO if discriminate order, we would have to unscramble here, and
                    # TODO then rescramble after getting hidden, before discriminator
                    inputs[k] = torch.cat([inputs[k], inputs[k][:, [0], :]], dim=1)

        batch_size = inputs["input_ids"].size(0)
        num_sent = inputs["input_ids"].size(1)

        if not unsupervised_training:
            labels = inputs.get("labels")
            del inputs["labels"]
        contrastive_labels = inputs.get("aug_labels").squeeze(1)
        del inputs["aug_labels"]

        # for discriminator
        if self.model_args.discriminate:
            aug_labels = inputs["aug_types"].squeeze(1)
        del inputs["aug_types"]

        # flatten input for encoding
        inputs["input_ids"] = inputs["input_ids"].view((-1, inputs["input_ids"].size(-1)))  # (bs * num_sent, len)
        inputs["attention_mask"] = inputs["attention_mask"].view(
            (-1, inputs["attention_mask"].size(-1))
        )  # (bs * num_sent len)
        if "token_type_ids" in inputs:  # not all model have this input
            inputs["token_type_ids"] = inputs["token_type_ids"].view(
                (-1, inputs["token_type_ids"].size(-1))
            )  # (bs * num_sent len)

        # forward pass
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        # TODO reminder this pooling may not be the same as original model pooling
        pooler_output = outputs.hidden_states[-1]
        logits = outputs.get("logits").view((batch_size, num_sent, model.config.num_labels))
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

        # Separate representation
        # Number of sentences in one instance
        # 2: sentence classification (OG, augmentation) or unsupervised (w/o label);
        # 3: sentence classification with positives (OG, augmentation, OG);
        # 4: pair sentence classification (OG1, augmentation, OG1+OG2, OG1+augmentation)
        # 5: pair sentence classification with positives (OG1, augmentation, OG1+OG2, OG1+augmentation, OG1)
        z_aug_pos = None
        if num_sent == 2:  # sentence classification
            z_orig, z_aug = pooler_output[:, 0], pooler_output[:, 1]
            logits_orig, logits_aug = logits[:, 0], logits[:, 1]
        elif num_sent == 4:  # pair sentence classification
            z_orig, z_aug = pooler_output[:, 0], pooler_output[:, 1]
            logits_orig, logits_aug = logits[:, 2], logits[:, 3]
        elif num_sent == 3:  # sentence classification with positives
            z_orig, z_aug, z_aug_pos = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]
            logits_orig, logits_aug = logits[:, 0], logits[:, 1]
        else:  # pair sentence classification with positives
            z_orig, z_aug, z_aug_pos = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 4]
            logits_orig, logits_aug = logits[:, 2], logits[:, 3]

        # Regular supervised loss
        if not unsupervised_training:
            if self.model.config.num_labels == 1:  # regression
                loss_fct = nn.MSELoss()
                supervised_orig_loss = loss_fct(logits_orig.squeeze(), labels.squeeze())
            else:  # classification
                loss_fct = nn.CrossEntropyLoss()
                supervised_orig_loss = loss_fct(logits_orig.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            supervised_orig_loss = torch.tensor(0.0).to(z_orig.device)
        model.supervised_orig_loss = supervised_orig_loss.detach().cpu().item()
        loss = supervised_orig_loss

        # Regular augmented supervised loss
        if self.model_args.supervised_augmentation_loss and not unsupervised_training:
            if self.model.config.num_labels == 1:  # regression
                supervised_aug_loss = loss_fct(logits_aug.squeeze(), labels.squeeze())
            else:  # classification
                # TODO add labeling option (if contrastive label = 1 and label = 1 them 1, etc)
                supervised_aug_loss = loss_fct(logits_aug.view(-1, self.model.config.num_labels), labels.view(-1))
            model.supervised_aug_loss = supervised_aug_loss.detach().cpu().item()
            loss += supervised_aug_loss
        else:
            model.supervised_aug_loss = 0.0

        # Contrastive loss for better sentence representation
        if self.model_args.contrastive_loss_type:
            if self.model_args.contrastive_loss_type in ["ranking_neg", "ranking_pos_neg"]:
                contrastive_loss = {
                    "ranking_pos_neg": self.multiple_negative_ranking_loss_with_positives,
                    "ranking_neg": self.multiple_negative_ranking_loss_no_in_batch_negatives,
                }[self.model_args.contrastive_loss_type](z_orig, z_aug_pos, z_aug)
            else:
                contrastive_loss = torch.tensor(0.0, device=z_orig.device)
                num_losses = len(self.model_args.contrastive_loss_type.split(","))
                for l_str in self.model_args.contrastive_loss_type.split(","):
                    if ":" in l_str:
                        l, weight = l_str.split(":")[0], float(l_str.split(":")[1])
                    else:
                        l, weight = l_str, 1.0 / num_losses
                    contrastive_loss += {
                        "ranking": self.multiple_negative_ranking_loss,
                        "triplet": self.triplet_loss,
                        "alignment": self.alignment_and_uniformity_loss,
                        "contrastive": self.contrastive_loss,
                        "hard_contrastive": self.hard_contrastive_loss,
                    }[l](z_orig, z_aug if not self.model_args.diff_cse_mode else z_aug_pos, contrastive_labels) * weight
            model.contrastive_loss = contrastive_loss.detach().cpu().item()

            # just for logging purpose, calculate alignment and uniformity loss as metrics
            with torch.no_grad():
                if self.model_args.contrastive_loss_type in ["ranking_pos_neg", "ranking_neg"]:
                    contrastive_labels = torch.ones_like(contrastive_labels).to(z_orig.device)
                self.alignment_and_uniformity_loss(
                    z_orig, z_aug, contrastive_labels, distance_func=DistanceMetric.COSINE, alpha=2, t=2, temp=1.0
                )
            loss += contrastive_loss
        else:
            model.contrastive_loss = 0.0

        # MLM loss or MoE loss
        loss += outputs.loss

        # discriminator loss
        if self.model_args.discriminate:
            aug_pred = model.discriminator(torch.cat([z_orig, z_aug], dim=1))
            num_augs = aug_pred.shape[1]
            discriminator_loss = F.binary_cross_entropy_with_logits(
                aug_pred, F.one_hot(aug_labels, num_classes=num_augs).float()
            )
            arg_max_pred = aug_pred.max(1)[1]
            model.discriminator_accuracy = ((arg_max_pred == aug_labels).sum() / batch_size).detach().cpu().item()
            model.discriminator_no_aug_predict_percent = (
                ((arg_max_pred == (num_augs - 1)).sum() / batch_size).detach().cpu().item()
            )
            model.discriminator_loss = discriminator_loss.detach().cpu().item()
            loss += discriminator_loss * self.model_args.discriminator_weight

        return (loss, outputs) if return_outputs else loss

    def compute_loss_pos_neg(self, model, inputs, return_outputs=False):
        unsupervised_training = True if "labels" not in inputs else False
        batch_size = inputs["input_ids"].size(0)
        num_sent = inputs["input_ids"].size(1)

        if not unsupervised_training:
            labels = inputs.get("labels")
            del inputs["labels"]

        # flatten input for encoding
        inputs["input_ids"] = inputs["input_ids"].view((-1, inputs["input_ids"].size(-1)))  # (bs * num_sent, len)
        inputs["attention_mask"] = inputs["attention_mask"].view(
            (-1, inputs["attention_mask"].size(-1))
        )  # (bs * num_sent len)
        if "token_type_ids" in inputs:  # not all model have this input
            inputs["token_type_ids"] = inputs["token_type_ids"].view(
                (-1, inputs["token_type_ids"].size(-1))
            )  # (bs * num_sent len)

        # forward pass
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True, return_dict=True)
        # TODO reminder this pooling may not be the same as original model pooling
        pooler_output = outputs.hidden_states[-1]
        logits = outputs.get("logits").view((batch_size, num_sent, model.config.num_labels))
        pooler_output = pooler_output.view((batch_size, num_sent, pooler_output.size(-1)))  # (bs, num_sent, hidden)

        # Separate representation
        # Number of sentences in one instance
        # 3: sentence classification (OG, neg aug, pos aug) or unsupervised (w/o label);
        # 6: pair sentence classification (OG1, neg aug, pos aug, OG1+OG2, neg aug+OG2, pos aug+OG2)
        if num_sent == 3:  # sentence classification or unsupervised
            z_orig, z_aug_neg, z_aug_pos = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]
            logits_orig, logits_aug_neg, logits_aug_pos = logits[:, 0], logits[:, 1], logits[:, 2]
        else:  # pair sentence classification with positives
            z_orig, z_aug_neg, z_aug_pos = pooler_output[:, 0], pooler_output[:, 1], pooler_output[:, 2]
            logits_orig, logits_aug_neg, logits_aug_pos = logits[:, 3], logits[:, 4], logits[:, 5]

        # Regular supervised loss
        if not unsupervised_training:
            if self.model.config.num_labels == 1:  # regression
                loss_fct = nn.MSELoss()
                supervised_orig_loss = loss_fct(logits_orig.squeeze(), labels.squeeze())
            else:  # classification
                loss_fct = nn.CrossEntropyLoss()
                supervised_orig_loss = loss_fct(logits_orig.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            supervised_orig_loss = torch.tensor(0.0).to(z_orig.device)
        model.supervised_orig_loss = supervised_orig_loss.detach().cpu().item()
        loss = supervised_orig_loss

        # Regular augmented supervised loss
        if self.model_args.supervised_augmentation_loss and not unsupervised_training:
            if self.model.config.num_labels == 1:  # regression
                supervised_aug_loss = loss_fct(logits_aug_pos.squeeze(), labels.squeeze())
                supervised_aug_loss += loss_fct(logits_aug_neg.squeeze(), labels.squeeze())
            else:  # classification
                # TODO add labeling option (if contrastive label = 1 and label = 1 them 1, etc)
                supervised_aug_loss = loss_fct(logits_aug_pos.view(-1, self.model.config.num_labels), labels.view(-1))
                supervised_aug_loss += loss_fct(logits_aug_neg.view(-1, self.model.config.num_labels), labels.view(-1))
            model.supervised_aug_loss = supervised_aug_loss.detach().cpu().item()
            loss += supervised_aug_loss
        else:
            model.supervised_aug_loss = 0.0

        # Contrastive loss for better sentence representation
        if self.model_args.contrastive_loss_type and len(z_orig) > 1:
            assert self.model_args.contrastive_loss_type in ["ranking_neg", "ranking_pos_neg"]
            contrastive_loss = {
                "ranking_pos_neg": self.multiple_negative_ranking_loss_with_positives,
                "ranking_neg": self.multiple_negative_ranking_loss_no_in_batch_negatives,
            }[self.model_args.contrastive_loss_type](z_orig, z_aug_pos, z_aug_neg)

            model.contrastive_loss = contrastive_loss.detach().cpu().item()

            # just for logging purpose, calculate alignment and uniformity loss as metrics
            with torch.no_grad():
                contrastive_labels = torch.ones(batch_size).to(z_orig.device)
                self.alignment_and_uniformity_loss(
                    z_orig, z_aug_pos, contrastive_labels, distance_func=DistanceMetric.COSINE, alpha=2, t=2, temp=1.0
                )
            loss += contrastive_loss
        else:
            model.contrastive_loss = 0.0

        # MLM or MoE loss
        loss += outputs.loss  # mlm loss weighted by mlm weight

        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False):
        # loss during evaluation on same dataset eval set (not SentEval)
        if "aug_labels" not in inputs and inputs["input_ids"].shape[1] == 1:
            inputs = {k: v.squeeze() for k, v in inputs.items()}
            return self.eval_supervised_loss(model, inputs, return_outputs)

        if self.args.use_pos_neg_pipeline:
            return self.compute_loss_pos_neg(model, inputs, return_outputs)
        else:
            return self.compute_loss_one_aug(model, inputs, return_outputs)


def get_dataset(data_args, training_args):
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column. You can easily tweak this
    # behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    data_files = {}
    if data_args.train_file is not None:
        data_files["train"] = data_args.train_file

    extension = data_args.train_file.split(".")[-1]
    if extension == "txt":
        extension = "text"
    if extension == "csv":
        datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=data_args.cache_dir,
            delimiter="\t" if "tsv" in data_args.train_file else ",",
        )
    else:
        datasets = load_dataset(extension, data_files=data_files, cache_dir=data_args.cache_dir)

    if training_args.eval_original:
        # load eval data
        eval_split = (
            "validation_matched"
            if data_args.dataset_name == "multi_nli"
            else "validation"
            if data_args.dataset_name != "imdb"
            else "test"
        )
        eval_dataset = get_huggingface_dataset(data_args.dataset_name, eval_split)
        datasets["eval"] = eval_dataset
    return datasets


def get_tokenizer(model_args):
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
    }
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, **tokenizer_kwargs)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return tokenizer


def get_model(tokenizer, logger, model_args, data_args):
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {"cache_dir": model_args.cache_dir, "num_labels": model_args.num_labels}
    if data_args.dataset_name in glue_task_to_keys:
        config_kwargs["finetuning_task"] = data_args.dataset_name

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if model_args.model_name_or_path:
        model = AutoModelForSequenceClassification.from_pretrained(model_args.model_name_or_path, config=config)
    else:
        raise NotImplementedError
        logger.info("Training new model from scratch")
        model = AutoModelForMaskedLM.from_config(config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pooler_type = model_args.pooler_type
    model.config.projection_layers = model_args.projection_layers
    model.config.projection_heads = model_args.projection_heads
    model.config.do_mlm = model_args.do_mlm
    model.config.mlm_weight = model_args.mlm_weight
    model.config.moe = model_args.moe
    model.config.moe_k = model_args.moe_k
    model.config.discriminate = model_args.discriminate
    model.config.gradient_reverse_multiplier = model_args.gradient_reverse_multiplier
    model.config.discriminator_dropout = model_args.discriminator_dropout
    model.config.discriminator_layers = model_args.discriminator_layers
    model.config.discriminate_order = model_args.discriminate_order
    model.config.discriminate_original = model_args.discriminate_original
    model.config.num_augmentations = len(data_args.augmentations)
    contrastive_model = ContrastiveModel(model)
    if model_args.do_mlm and "bert" in model_args.model_name_or_path:
        pretrained_model = BertForPreTraining.from_pretrained(model_args.model_name_or_path)
        contrastive_model.lm_head.load_state_dict(pretrained_model.cls.predictions.state_dict())
    return contrastive_model


def prepare_train_feature_one_aug_batch(
    datasets: Dict[str, Dataset], tokenizer, data_args, epoch: Optional[float] = 0
) -> Dataset:
    """
    Old training preprocessing pipeline, resulting dataset contains an additional sent1_aug column along with
    the aug_labels column that indicate whether the augmentation is positive or negative
    """
    train_dataset = datasets["train"]
    num_augs = len(data_args.augmentations)
    og_dataset_len = int(len(train_dataset) / num_augs)
    print(
        f"One Aug Feature Pipeline: unprocessed dataset aggregate length={len(train_dataset)}, #augs={num_augs},"
        f" og dataset length={og_dataset_len}"
    )
    aug_index = np.tile(np.arange(num_augs), [og_dataset_len, 1]).T.reshape(-1)
    # subsample / recombing different augmentations
    if len(data_args.augmentations) > 0:
        # for every row, take 2 samples with no replacement,
        # https://stackoverflow.com/questions/53891169/numpy-take-many-samples-with-no-replacement-by-row
        sample_indices = np.random.rand(og_dataset_len, num_augs).argpartition(2, axis=1)[:, :2] * og_dataset_len
        sample_indices += np.tile(np.arange(og_dataset_len), [2, 1]).T
        aug1 = train_dataset.select(sample_indices[:, 0])["sent1_aug"]
        aug2 = train_dataset.select(sample_indices[:, 1])["sent1_aug"]
        # TODO may cause issue when expanding to supervised settings
        train_dataset = Dataset.from_dict({"sent1": aug1, "sent1_aug": aug2, "aug_labels": [1] * og_dataset_len})

    elif data_args.uniform_augmentation_sampling and len(data_args.augmentations) > 0:
        # get original dataset size
        sample_index = np.arange(og_dataset_len)
        # randomly pick an augmentation (scale up by dataset size, then we get index size. This assumes we don't remove
        # no augmentation indices and every augmentation returns same length dataset
        aug_index = np.random.randint(0, num_augs, size=len(sample_index))
        sample_index += aug_index * len(sample_index)
        train_dataset = train_dataset.select(sample_index)

    # add augmentation type label
    train_dataset = train_dataset.add_column("aug_types", aug_index)

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), data_args.max_train_samples))

    # Prepare features
    column_names = sorted([n for n in datasets["train"].column_names if n in ["sent1", "sent1_aug", "sent2"]])

    if len(column_names) == 3:
        # Pair datasets with augmentation
        sent1_cname = column_names[0]
        sent1_aug_cname = column_names[1]
        sent2_cname = column_names[2]
    elif len(column_names) == 2:
        # Single sentence classification datasets with augmentation
        sent1_cname = column_names[0]
        sent1_aug_cname = column_names[1]
    elif len(column_names) == 1:
        # Unsupervised datasets, and no augmentation
        # column_names.append(column_names[0])
        sent1_cname = column_names[0]
        sent1_aug_cname = column_names[0]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[column_names[0]])

        # Avoid "None" fields
        for idx in range(total):
            for c_name in column_names:
                if examples[c_name][idx] is None:
                    examples[c_name][idx] = ""

            if examples[sent1_cname][idx] == examples[sent1_aug_cname][idx] or examples[sent1_aug_cname][idx] == "":
                examples["aug_types"][idx] = num_augs  # last label reserved for no augmentation

        # if only discriminate aug vs no aug, binarize labels
        if data_args.discriminate_original:
            for idx in range(total):
                examples["aug_types"][idx] = 1 if examples["aug_types"][idx] == num_augs else 0

        sentences = examples[sent1_cname] + examples[sent1_aug_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        if len(column_names) == 3:
            sentence_pair_features = tokenizer(
                examples[sent1_cname],
                examples[sent2_cname],
                max_length=data_args.max_seq_length,
                truncation=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            aug_sentence_pair_features = tokenizer(
                examples[sent1_aug_cname],
                examples[sent2_cname],
                max_length=data_args.max_seq_length,
                truncation=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            for key in sent_features:
                features[key] = [
                    [
                        sent_features[key][i],
                        sent_features[key][i + total],
                        sentence_pair_features[key][i],
                        aug_sentence_pair_features[key][i],
                    ]
                    for i in range(total)
                ]
        else:
            for key in sent_features:
                features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]

        if data_args.discriminate_order:
            swap_idx = np.random.choice([True, False], size=total, p=[0.5, 0.5])
            examples["aug_types"] = swap_idx.astype(int).tolist()

            sent1_tmp, aug_tmp = np.array(examples[sent1_cname]), np.array(examples[sent1_aug_cname])
            sent1_tmp[swap_idx], aug_tmp[swap_idx] = aug_tmp[swap_idx], sent1_tmp[swap_idx]
            examples[sent1_cname], examples[sent1_aug_cname] = sent1_tmp, sent1_aug_cname

        return features

    remove_cols = [c for c in train_dataset.column_names if c not in ["labels", "aug_labels", "aug_types"]]
    train_dataset = train_dataset.map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=remove_cols,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if data_args.remove_no_augmentations and len(data_args.augmentations) > 0:
        train_dataset = train_dataset.filter(lambda example: example["aug_types"] == num_augs)

    # wandb.run.summary["pos_augs"] = sum(train_dataset["aug_labels"])
    # wandb.run.summary["neg_augs"] = len(train_dataset) - wandb.run.summary["pos_augs"]
    # wandb.run.summary["pos_augs_percent"] = round(wandb.run.summary["pos_augs"] / len(train_dataset), 4)
    # wandb.run.summary["neg_augs_percent"] = round(wandb.run.summary["neg_augs"] / len(train_dataset), 4)
    wandb.run.summary["no_aug_percent"] = round(
        (np.array(train_dataset["aug_types"]) == num_augs).sum() / len(train_dataset), 4
    )
    wandb.run.summary["train_length"] = len(train_dataset)
    return train_dataset


def prepare_train_feature_pos_neg_batch(
    datasets: Dict[str, Dataset], tokenizer, data_args, epoch: Optional[float] = 0
) -> Dataset:
    """
    New training preprocessing pipeline which combines multiple positive and negative augmentations into
    one compact dataset with same length as the original dataset. It uses default positive augmentation
    and default negative augmentation to ensure all examples have both augmentations. Out of all other
    augmentations, the pipeline chooses the augmentation uniformly.
    """
    train_dataset = datasets["train"]

    # combine multiple augmentations
    # get original dataset size
    num_augs = len(data_args.augmentations)
    dataset_len = int(len(train_dataset) / len(data_args.augmentations))
    print(f"training dataset initial full length: {len(train_dataset)}")
    print(f"training dataset length: {dataset_len}")

    pos_aug = np.array(train_dataset["sent1"][:dataset_len], dtype=object)  # default is original sentence (SimCSE)
    aug = RandomDeletion(prob=0.6, aug_max=30)
    neg_aug = np.array([aug.generate(s)[0] for s in pos_aug], dtype=object)  # need to have good default negative aug

    # create index of successful augmentation and use it to filter out indices
    tmp_dataset = train_dataset.map(
        lambda example: {"success_idx": example["sent1"] != example["sent1_aug"] and example["sent1_aug"] != ""}
    )
    success_idx = np.array(tmp_dataset["success_idx"]).reshape(num_augs, dataset_len).T
    del tmp_dataset
    aug_labels = np.array(train_dataset["aug_labels"]).reshape(num_augs, dataset_len).T

    pos_tgt_idx = np.logical_and(aug_labels, success_idx).sum(axis=1) >= 1
    neg_tgt_idx = np.logical_and(np.logical_not(aug_labels), success_idx).sum(axis=1) >= 1
    if data_args.sample_default_augmentations:
        # randomly remove 1/(#augmentations+1) of idx to keep as default
        pos_tgt_idx = np.logical_and(
            pos_tgt_idx,
            np.random.choice([True, False], size=dataset_len, p=[1 - (1 / (num_augs + 1)), 1 / (num_augs + 1)]),
        )
        neg_tgt_idx = np.logical_and(
            neg_tgt_idx,
            np.random.choice([True, False], size=dataset_len, p=[1 - (1 / (num_augs + 1)), 1 / (num_augs + 1)]),
        )

    # randomly select the source index from where the augmented label will be copied from
    pos_srs_idx = np.array([random.choice((row == 1).nonzero()[0]) for row in aug_labels[pos_tgt_idx]]) * dataset_len
    neg_srs_idx = np.array([random.choice((row == 0).nonzero()[0]) for row in aug_labels[neg_tgt_idx]]) * dataset_len
    pos_srs_idx += np.arange(dataset_len)[pos_tgt_idx]
    neg_srs_idx += np.arange(dataset_len)[neg_tgt_idx]

    if data_args.ablate_positive_augmentations:
        pos_srs_idx = np.array([])

    if data_args.ablate_negative_augmentations:
        neg_srs_idx = np.array([])

    # copy from source index to target index
    if len(pos_srs_idx) > 0:
        pos_aug[pos_tgt_idx] = np.array(train_dataset.select(pos_srs_idx)["sent1_aug"], dtype=object)
    if len(neg_srs_idx) > 0:
        neg_aug[neg_tgt_idx] = np.array(train_dataset.select(neg_srs_idx)["sent1_aug"], dtype=object)

    compact_train_dataset = train_dataset.select(np.arange(dataset_len))
    compact_train_dataset = compact_train_dataset.add_column("sent1_aug_pos", pos_aug)
    compact_train_dataset = compact_train_dataset.add_column("sent1_aug_neg", neg_aug)
    compact_train_dataset = compact_train_dataset.remove_columns(["aug_labels", "sent1_aug"])
    train_dataset = compact_train_dataset

    if data_args.remove_no_augmentations and len(data_args.augmentations) > 0:
        # works for single augmentation but not multiple, but this works with not changing aug label
        # train_dataset = train_dataset.select(np.arange(dataset_len)[success_idx.squeeze()])
        train_dataset = train_dataset.select(np.arange(dataset_len)[neg_tgt_idx])

    if data_args.max_train_samples is not None:
        train_dataset = train_dataset.select(random.sample(range(len(train_dataset)), data_args.max_train_samples))

    # Prepare features
    column_names = sorted(
        [n for n in train_dataset.column_names if n in ["sent1", "sent1_aug_neg", "sent1_aug_pos", "sent2"]]
    )
    if len(column_names) == 4:
        # Pair datasets with augmentation
        sent1_cname = column_names[0]
        sent1_neg_cname = column_names[1]
        sent1_pos_cname = column_names[2]
        sent2_cname = column_names[3]
    elif len(column_names) == 3:
        # Single sentence classification datasets with augmentation or
        # Unsupervised datasets, with positive and negative augmentation
        sent1_cname = column_names[0]
        sent1_neg_cname = column_names[1]
        sent1_pos_cname = column_names[2]
    else:
        raise NotImplementedError

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[column_names[0]])

        # Avoid "None" fields
        for idx in range(total):
            for c_name in column_names:
                if examples[c_name][idx] is None:
                    examples[c_name][idx] = " "

        sentences = examples[sent1_cname] + examples[sent1_neg_cname] + examples[sent1_pos_cname]

        sent_features = tokenizer(
            sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        features = {}
        if len(column_names) == 4:
            sentence_pair_features = tokenizer(
                examples[sent1_cname],
                examples[sent2_cname],
                max_length=data_args.max_seq_length,
                truncation=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            pos_aug_sentence_pair_features = tokenizer(
                examples[sent1_pos_cname],
                examples[sent2_cname],
                max_length=data_args.max_seq_length,
                truncation=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            neg_aug_sentence_pair_features = tokenizer(
                examples[sent1_neg_cname],
                examples[sent2_cname],
                max_length=data_args.max_seq_length,
                truncation=True,
                padding="max_length" if data_args.pad_to_max_length else False,
            )
            for key in sent_features:
                features[key] = [
                    [
                        sent_features[key][i],
                        sent_features[key][i + total],
                        sent_features[key][i + total * 2],
                        sentence_pair_features[key][i],
                        neg_aug_sentence_pair_features[key][i],
                        pos_aug_sentence_pair_features[key][i],
                    ]
                    for i in range(total)
                ]
        else:
            for key in sent_features:
                features[key] = [
                    [sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]]
                    for i in range(total)
                ]

        return features

    remove_cols = [c for c in train_dataset.column_names if c not in ["labels", "aug_labels"]]
    train_dataset = train_dataset.map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=remove_cols,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    data_stat = {
        "epoch": epoch,
        "pos_augs": len(pos_srs_idx),
        "neg_augs": len(neg_srs_idx),
        "train_length": len(train_dataset),
    }
    data_stat["pos_augs_percent"] = round(data_stat["pos_augs"] / dataset_len, 4)
    data_stat["neg_augs_percent"] = round(data_stat["neg_augs"] / dataset_len, 4)
    wandb.log(data_stat)

    return train_dataset


def prepare_eval_feature(datasets: Dict[str, Dataset], tokenizer, data_args) -> Dataset:
    # Prepare features
    if datasets["eval"].config_name in glue_task_to_keys:
        column_names = glue_task_to_keys[datasets["eval"].config_name]
    else:
        column_names = [c for c in datasets["eval"].column_names if not "label" in c]
        if "idx" in column_names:
            column_names.remove("idx")

    def prepare_features(examples):
        # padding = longest (default)
        #   If no sentence in the batch exceed the max length, then use
        #   the max sentence length in the batch, otherwise use the
        #   max sentence length in the argument and truncate those that
        #   exceed the max length.
        # padding = max_length (when pad_to_max_length, for pressure test)
        #   All sentences are padded/truncated to data_args.max_seq_length.
        total = len(examples[column_names[0]])

        # Avoid "None" fields
        for idx in range(total):
            for c_name in column_names:
                if examples[c_name][idx] is None:
                    examples[c_name][idx] = " "

        sentences = (
            (examples[column_names[0]],)
            if len(column_names) == 1
            else (examples[column_names[0]], examples[column_names[1]])
        )

        sent_features = tokenizer(
            *sentences,
            max_length=data_args.max_seq_length,
            truncation=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )

        return sent_features

    remove_column_names = [c for c in datasets["eval"].column_names if not "label" in c]

    eval_dataset = datasets["eval"].map(
        prepare_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=remove_column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )
    if data_args.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(random.sample(range(len(eval_dataset)), data_args.max_eval_samples))
    return eval_dataset


def get_compute_metrics(raw_datasets, data_args) -> Tuple[Optional[Callable], int]:
    # Labels
    label_col = "label" if "label" in raw_datasets["train"].features else "labels"
    if label_col not in raw_datasets["train"].features:
        return None, 1

    is_regression = raw_datasets["train"].features[label_col].dtype in ["float32", "float64"]

    if is_regression:
        num_labels = 1
    else:
        # A useful fast method:
        # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
        label_list = raw_datasets["train"].unique(label_col)
        label_list.sort()  # Let's sort it for determinism
        num_labels = len(label_list)

    # Get the metric function
    try:
        metric = load_metric("glue", data_args.dataset_name)
    except:
        metric = load_metric("accuracy")

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.dataset_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    return compute_metrics, num_labels


def init_wandb(data_args, model_args, training_args):
    wandb.init(
        project=training_args.wandb_project,
        name=training_args.run_name,
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.config.update(data_args.to_dict())
    wandb.config.update(model_args.to_dict())


def train(
    model_args,
    data_args,
    training_args,
    logger,
):
    init_wandb(data_args, model_args, training_args)

    datasets = get_dataset(data_args, training_args)
    compute_metrics, num_labels = get_compute_metrics(datasets, data_args)
    model_args.num_labels = num_labels
    tokenizer = get_tokenizer(model_args)
    model = get_model(tokenizer, logger, model_args, data_args)
    if training_args.use_pos_neg_pipeline:
        # new processing pipeline, positive and negative aug for each example
        train_dataset = prepare_train_feature_pos_neg_batch(datasets, tokenizer, data_args)
    else:
        train_dataset = prepare_train_feature_one_aug_batch(datasets, tokenizer, data_args)
    eval_dataset = prepare_eval_feature(datasets, tokenizer, data_args) if training_args.eval_original else None
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else OurDataCollatorWithPadding(tokenizer, do_mlm=model_args.do_mlm)
    )
    training_args.data_args = data_args

    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[TrainingDataTransformationCallback],
    )

    if isinstance(model_args.distance_metric, str):
        model_args.distance_metric = getattr(DistanceMetric, model_args.distance_metric)

    trainer.model_args = model_args
    model_path = (
        model_args.model_name_or_path
        if (model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path))
        else None
    )
    if training_args.do_train:

        train_result = trainer.train(model_path=model_path)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(train_result.metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            max_train_samples = (
                data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
            )
            train_result.metrics["train_samples"] = min(max_train_samples, len(train_dataset))
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # evaluation TODO
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        results = trainer.evaluate(eval_senteval_transfer=True, eval_robust=True)
        # results = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in sorted(results.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

        # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        # results["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        trainer.log_metrics("eval", results)
        trainer.save_metrics("eval", results)

    return results
