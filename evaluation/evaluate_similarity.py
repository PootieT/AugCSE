from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from typing import Tuple, Dict, List, Union, Any, Callable, Optional

from interfaces.Operation import Operation
from tasks.TaskTypes import TaskType
import numpy as np
import enum

from datasets import load_dataset
from transformers import pipeline

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


glue_task_to_keys = {
    "cola": ["sentence"],
    "mnli": ["premise", "hypothesis"],
    "mrpc": ["sentence1", "sentence2"],
    "qnli": ["question", "sentence"],
    "qqp": ["question1", "question2"],
    "rte": ["sentence1", "sentence2"],
    "sst2": ["sentence"],
    "stsb": ["sentence1", "sentence2"],
    "wnli": ["sentence1", "sentence2"],
}


def get_huggingface_dataset(dataset_name: str, split: str):
    if dataset_name in glue_task_to_keys:
        hf_dataset = load_dataset("glue", dataset_name, split=split)
    elif dataset_name in ["clue"]:
        hf_dataset = load_dataset(dataset_name, "cluewsc2020", split=split)
    elif dataset_name.endswith(".csv") or dataset_name.endswith(".txt"):
        # load local dataset
        dataset_name_split = dataset_name.split("/")
        data_path = "/".join(dataset_name_split[:-1])
        data_files = {"train": dataset_name_split[-1]}
        hf_dataset = load_dataset(data_path, data_files=data_files, split=split)
    else:
        hf_dataset = load_dataset(dataset_name, split=split)
    return hf_dataset


def _process_data(dataset_name, split, max_size: Optional[int] = 1000, load_perturb=False) -> BaseDataset:
    hf_dataset = get_huggingface_dataset(dataset_name, split)

    if dataset_name.endswith(".csv") or dataset_name.endswith(".txt"):  # local file
        dataset_name = dataset_name.split("/")[-1].split("_")[0]

    if dataset_name == "imdb" or "imdb" in dataset_name:
        label_name = "label"
        instance_name = ["text"]
        data_class = TextLineDataset
    elif dataset_name == "sst2" or "sst2" in dataset_name:
        label_name = "label"
        instance_name = ["sentence"]
        data_class = TextLineDataset
    elif dataset_name == "clue" or "clue" in dataset_name:
        label_name = "label"
        instance_name = ["text"]
        data_class = TextLineDataset
    elif dataset_name in ["mnli", "snli"] or "nli" in dataset_name:
        label_name = "label"

        instance_name = ["premise", "hypothesis"]
        data_class = KeyValueDataset
    elif dataset_name == "qqp" or "qqp" in dataset_name:
        label_name = "label"
        instance_name = ["question1", "question2"]

        data_class = KeyValueDataset
    elif dataset_name == "wiki1m":
        label_name = None
        instance_name = ["text"] if not load_perturb else ["sent1_aug"]
        data_class = UnsupervisedTextDataset
    else:
        label_name = "label"
        instance_name = glue_task_to_keys[dataset_name]
        data_class = KeyValueDataset

    datasets = data_class.from_huggingface(
        hf_dataset,
        fields=instance_name + [label_name] if label_name is not None else instance_name,
        task_type=TaskType.TEXT_CLASSIFICATION,
        max_size=max_size,
        # max_size=80,  # Debug purpose only
    )
    return datasets


def _get_instance_by_keys(example):
    if type(example) == str:
        return example
    elif len(example) == 1:
        return example[0] if type(example[0]) == str else example[0][0]
    else:
        # if a tuple (QQP, MNLI), return first sentence only (only one perturbed)
        return [e if type(e) == str else e[0] for e in example][0]


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
    # feature_pipeline = pipeline(
    #     "feature-extraction",
    #     model=model_name,
    #     tokenizer=model_name,
    #     device=0 if is_cuda else -1,
    # )
    sentence_transformer = SentenceTransformer(model_name)

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
        pt_dataset = _process_data(
            pt_dataset_path, split, max_size=None if "train" in split else 1000, load_perturb=True
        )
    else:
        pt_dataset = dataset.apply_transformation(operation, batch_size=augmentation_batch_size)
        performance["perturbation_rate"] = pt_dataset.perturbation_rate
    print("Here is the cosine similarity of the sentence embeddings between augmented and original datasets")
    similarities, similarity_samples = evaluate_dataset(
        sentence_transformer, dataset, pt_dataset, batch_size=batch_size
    )
    performance.update(similarities)
    # (3) Execute perturbation
    # (4) Execute the similarity of the original set and the perturbed set
    print(f"Similarity ={performance}")
    if return_dataset:
        return performance, similarity_samples, dataset, pt_dataset
    else:
        return performance, similarity_samples


def get_sentence_embedding(model, model_input, pooling: str = "mean") -> torch.Tensor:
    # source: https://huggingface.co/sentence-transformers/bert-base-nli-mean-tokens
    # Mean Pooling - Take attention mask into account for correct averaging
    model_output = model(model_input, truncation=True, batch_size=len(model_input))
    if pooling == "mean":
        agg_output = torch.vstack([torch.tensor(doc[0]).mean(dim=0) for doc in model_output])
    else:
        raise (f"Pooling method {pooling} not implemented.")

    return agg_output


def similarity(x, y, temp=1.0):
    cos = nn.CosineSimilarity(dim=-1)
    return cos(x, y) / temp


def sample_non_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Given a d X d square matrix, randomly sample d entries from the non-diagonals"""
    d = x.shape[0]
    idx = torch.randperm(d * d - d)
    return x.masked_select(~torch.eye(d, dtype=bool, device=x.device))[idx][:d]


def _get_model_pred(
    model, examples: List[str], pt_examples: List[str], batch_size: int
) -> Tuple[Dict[str, float], Dict[str, np.array]]:
    similarities = []
    similarity_samples = []
    d = "cuda" if torch.cuda.is_available() else "cpu"
    sim_bs = 256
    with torch.no_grad():
        for e in tqdm((range(0, len(examples), sim_bs)), position=0, leave=True):
            orig_emb = model.encode(
                examples[e : e + sim_bs], batch_size=batch_size, device="cuda", convert_to_tensor=True
            )
            aug_emb = model.encode(
                pt_examples[e : e + sim_bs], batch_size=batch_size, device="cuda", convert_to_tensor=True
            )
            if len(orig_emb) != len(aug_emb):
                continue  # for some wacky reason
            orig_sim = similarity(orig_emb.unsqueeze(1), orig_emb.unsqueeze(0))
            aug_sim = similarity(aug_emb.unsqueeze(1), aug_emb.unsqueeze(0))
            cross_sim = similarity(orig_emb.unsqueeze(1), aug_emb.unsqueeze(0))

            actual_bs = len(orig_sim)
            # average similarity between a sample and a random sample
            orig_rand_sim = torch.sum(orig_sim - orig_sim * torch.eye(actual_bs, device=d)) / (
                actual_bs ** 2 - actual_bs
            )
            # randomly select batch size number of pairs of similarities other than diagonal ones
            orig_rand_sim_sample = sample_non_diagonal(orig_sim)

            # average similarity between a augmented sample and a random augmented sample
            aug_rand_sim = torch.sum(aug_sim - aug_sim * torch.eye(actual_bs, device=d)) / (actual_bs ** 2 - actual_bs)
            aug_rand_sim_sample = sample_non_diagonal(aug_sim)

            # average similarity between the original and the augmented sample
            aug_self_sim = torch.sum(torch.diagonal(cross_sim, 0)) / actual_bs
            aug_self_sim_sample = torch.diagonal(cross_sim, 0)

            # average similarity between a sample and a random augmented sample
            aug_orig_rand_sim = torch.sum(cross_sim - cross_sim * torch.eye(actual_bs, device=d)) / (
                actual_bs ** 2 - actual_bs
            )
            aug_orig_rand_sim_sample = sample_non_diagonal(cross_sim)

            similarities.append([orig_rand_sim, aug_rand_sim, aug_self_sim, aug_orig_rand_sim])
            similarity_samples.append(
                [
                    orig_rand_sim_sample,
                    aug_rand_sim_sample,
                    aug_self_sim_sample,
                    aug_orig_rand_sim_sample,
                ]
            )

    similarities = torch.hstack([torch.vstack([i[0], i[1], i[2], i[3]]) for i in similarities]).mean(1)
    similarity_samples = (
        torch.hstack([torch.vstack([i[0], i[1], i[2], i[3]]) for i in similarity_samples]).cpu().numpy()
    )
    similarities_dict = {
        "orig_rand_sim": similarities[0].item(),
        "aug_rand_sim": similarities[1].item(),
        "aug_self_sim": similarities[2].item(),
        "aug_orig_rand_sim": similarities[3].item(),
    }
    similarity_samples_dict = {
        "orig_rand_sim_sample": similarity_samples[0],
        "aug_rand_sim_sample": similarity_samples[1],
        "aug_self_sim_sample": similarity_samples[2],
        "aug_orig_rand_sample": similarity_samples[3],
    }
    return similarities_dict, similarity_samples_dict


def evaluate_dataset(
    sentence_transformer, dataset, pt_dataset, batch_size=32
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

    assert len(examples) == len(pt_examples)
    similarities, similarity_samples = _get_model_pred(
        sentence_transformer,
        examples,
        pt_examples,
        batch_size=batch_size,
    )
    print(f"The similarities on this subset which has {len(examples)} examples: {similarities}")
    return similarities, similarity_samples
