from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from typing import Tuple, Dict, List, Any, Optional, Union

from evaluation.evaluate_similarity import _process_data, _get_instance_by_keys
from interfaces.Operation import Operation
from tasks.TaskTypes import TaskType
import numpy as np
import enum

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from dataset import TextLineDataset, KeyValueDataset, BaseDataset, UnsupervisedTextDataset
import torch
import torch.nn as nn

# make this to work for three task.


class SENTIMENT_LABELS(enum.Enum):
    NEGATIVE = 0
    POSITIVE = 1


class NLI_LABELS(enum.Enum):
    ENTAILMENT = 0
    NEUTRAL = 1
    CONTRADICTION = 2


class QQP_LABEL(enum.Enum):
    NON_DUPLICATE = 0
    DUPLICATE = 1


def evaluate(
    operation: Operation,
    model_name: str,
    dataset_name: str,
    split="test[:20%]",
    batch_size=8,
    is_cuda=torch.cuda.is_available(),
    pt_dataset_path: Optional[str] = None,
    return_dataset: bool = False,
    augmentation_batch_size: Optional[int] = None,
) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Tuple[Dict[str, float], Dict[str, Any], BaseDataset, BaseDataset],]:
    if model_name is None:
        model_name = "aychang/roberta-base-imdb"
    if dataset_name is None:
        dataset_name = "imdb"
    print(f"Loading <{dataset_name}> dataset to evaluate <{model_name}> model.")
    # is_cuda = False
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device="cuda:0" if is_cuda else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    percent = f"[{split.split('[')[-1]}" if "[" in split else ""
    if dataset_name == "mnli" and "train" not in split:
        split = f"validation_matched{percent}"
    elif dataset_name != "imdb" and "train" not in split:
        split = f"validation{percent}"

    performance = {
        "model_name": model_name,
        "split": split,
        "dataset_name": dataset_name,
    }
    dataset = _process_data(dataset_name, split, max_size=None if "train" in split else 1000)

    if pt_dataset_path:
        # load an existing perturbed dataset
        pt_dataset = _process_data(pt_dataset_path, split, max_size=None if "train" in split else 1000)
    else:
        # generate a perturbed dataset with transformation
        pt_dataset = dataset.apply_transformation(operation, batch_size=augmentation_batch_size)
        performance["perturbation_rate"] = pt_dataset.perturbation_rate
    print("Here is the perplexity of the sentence in augmented and original datasets")
    ppl, ppl_samples = evaluate_dataset(
        model,
        tokenizer,
        dataset,
        pt_dataset,
        model_name,
        batch_size=batch_size,
    )
    performance.update(ppl)
    # (3) Execute perturbation
    # (4) Execute the perplexity of the original set and the perturbed set
    print(f"Perplexity ={performance}")
    if return_dataset:
        return performance, ppl_samples, dataset, pt_dataset
    else:
        return performance, ppl_samples


def get_perplexity(model, tokenizer, model_input: List[str], stride: int = 512) -> torch.Tensor:
    # source: https://huggingface.co/docs/transformers/perplexity
    max_length = model.config.n_positions
    stride = min(max_length, stride)

    encodings = tokenizer(model_input, return_tensors="pt", padding="longest", truncation=True)

    nlls = []
    # if sentences are longer than default window size
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        # padded tokens set to -100, no attention no loss https://github.com/huggingface/transformers/issues/2630
        target_ids[target_ids == tokenizer.vocab[tokenizer.pad_token]] = -100
        # switch it to EOS because model word embedding doesn't have EOS. As long as label is -100 what token it
        # switches to doesn't impact performance
        input_ids[input_ids == tokenizer.vocab[tokenizer.pad_token]] = tokenizer.eos_token_id

        with torch.no_grad():
            # instead of taking aggregated cross entropy from causal LM, we calculate
            # per sentence without reduction.
            # https://github.com/huggingface/transformers/blob/v4.17.0/src/transformers/models/gpt2/modeling_gpt2.py#L1072
            outputs = model(input_ids)
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.view(shift_labels.shape).sum(dim=1)

        nlls.append(neg_log_likelihood)

    sent_lens = encodings.attention_mask.sum(dim=1).to(model.device)
    ppl = torch.exp(torch.stack(nlls).sum(dim=0) / sent_lens)
    return ppl


def similarity(x, y, temp=1.0):
    cos = nn.CosineSimilarity(dim=-1)
    return cos(x, y) / temp


def _get_model_pred(
    model,
    tokenizer,
    examples: List[str],
    pt_examples: List[str],
    batch_size: int,
) -> Tuple[Dict[str, float], Dict[str, np.array]]:
    orig_perplexities = np.array([])
    aug_perplexities = np.array([])

    with torch.no_grad():
        for e in tqdm((range(0, len(examples), batch_size)), leave=True, position=0):
            orig_ppl = get_perplexity(model, tokenizer, examples[e : e + batch_size])
            aug_ppl = get_perplexity(model, tokenizer, pt_examples[e : e + batch_size])

            orig_perplexities = np.append(orig_perplexities, orig_ppl.cpu().numpy())
            aug_perplexities = np.append(aug_perplexities, aug_ppl.cpu().numpy())

    ppl_dict = {
        "orig_ppl": float(np.mean(orig_perplexities)),
        "aug_ppl": float(np.mean(aug_perplexities)),
        "diff_ppl": float(np.mean(aug_perplexities - orig_perplexities)),
    }
    ppl_samples_dict = {
        "orig_ppl_sample": orig_perplexities,
        "aug_ppl_sample": aug_perplexities,
        "diff_ppl_sample": aug_perplexities - orig_perplexities,
    }
    return ppl_dict, ppl_samples_dict


def evaluate_dataset(
    model, tokenizer, dataset, pt_dataset, model_name, batch_size=32
) -> Tuple[Dict[str, float], Dict[str, np.array]]:
    if isinstance(dataset, UnsupervisedTextDataset):
        examples = [raw_text for raw_text in dataset]
        pt_examples = [raw_text for raw_text in pt_dataset]
    else:
        examples = [_get_instance_by_keys(list(raw_text)[:-1]) for raw_text in dataset]
        pt_examples = [_get_instance_by_keys(list(raw_text)[:-1]) for raw_text in pt_dataset]

    # clean out nan or wacky datapoints
    examples = [s if isinstance(s, str) else " " for s in examples]
    pt_examples = [s if isinstance(s, str) else " " for s in pt_examples]

    ppl, ppl_samples = _get_model_pred(model, tokenizer, examples, pt_examples, batch_size=batch_size)
    print(f"The perplexity on this subset which has {len(examples)} examples: {ppl}")
    return ppl, ppl_samples
