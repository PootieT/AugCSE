import hashlib
import os.path
import pathlib
import json
from typing import List, Optional, Union

import numpy as np
import pandas as pd

from dataset import UnsupervisedTextDataset, TextLineDataset
from evaluation import evaluate_similarity, evaluate_perplexity
from evaluation.evaluate_similarity import _get_instance_by_keys, _process_data
from interfaces.Operation import Operation
from training.utils import normalize_dataset_names

project_root = pathlib.Path(__file__).parent.resolve().parent


def hash_str(s: str) -> str:
    hash_int = int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16) % 10 ** 8
    return str(hash_int)


def label_with_semantic_preservation_and_lm_uniform_threshold(
    df: pd.DataFrame,
    operation: Operation,
    perplexity_std_scalar: float = 1.0,
    similarity_std_scalar: int = 0.2,
    return_stat: bool = False,
) -> pd.DataFrame:
    """
    Label augmentation based on whether operation is meaning preserving and langauge model outputs.
    If operation is labeled 'meaning-alteration', label negative.
    If operation is labeled 'highly-meaning-preserving', label positive.
    If operation is neither:
        If perplexity is too far off from original, label negative
        if sentence similarity is within batch random similarity distribution, label negative
        otherwise label positive
    :param df: dataframe with data instances and attributes
    :param operation: operation / transformation being used
    :param perplexity_std_scalar: how many standard deviation away from mean to consider a ppl out of distribution
    :param similarity_std_scalar: how many standard deviation away from mean to consider a sim out of distribution
    :param return_stat: if return filter stat based on perplexity and similarity
    """
    df["aug_labels"] = True
    if "highly-meaning-preserving" in operation.keywords:
        df["aug_labels"] = True
    elif "meaning-alteration" in operation.keywords:
        df["aug_labels"] = False
    else:
        # assume perplexity follows Gaussian distribution
        # perplexity standard of comparison have to be dataset specific and not dependent on the
        # augmentation. Therefore, we set the same standards using dataset perplexity distribution,
        # and we compare data point difference to the distribution because ultimately, we need to
        # filter out examples with larger difference in ppl
        ppl_mean, ppl_std = (
            df["orig_ppl_sample"].mean(),
            df["orig_ppl_sample"].std(),
        )
        # if augmented perplexity is too far off original perplexity distribution, negative
        ppl_filter_idx = (df["diff_ppl_sample"]) > perplexity_std_scalar * ppl_std
        df.loc[ppl_filter_idx, "aug_labels"] = False
        ppl_filter_perc = sum(ppl_filter_idx) / len(df) * 100
        print(
            f"{sum(ppl_filter_idx)} ({ppl_filter_perc:.4f}%) "
            f"of samples assigned negative due to high perplexity difference"
        )

        # assume similarity follows Gaussian distribution
        # again threshold should not be dependent on augmentation, only on dataset
        sim_mean, sim_std = (
            df["orig_rand_sim_sample"].mean(),
            df["orig_rand_sim_sample"].std(),
        )
        # if augmented similarity is indistinguishable from original similarity distribution, negative
        sim_filter_idx = (df["aug_self_sim_sample"] - df["orig_rand_sim_sample"]) < similarity_std_scalar * sim_std
        # sim_filter_idx = (1 - df["aug_self_sim_sample"]) > similarity_std_scalar * sim_std
        df.loc[sim_filter_idx, "aug_labels"] = False
        sim_filter_perc = sim_filter_idx.sum() / len(df) * 100
        print(
            f"{sum(sim_filter_idx)} ({sim_filter_perc:.2f}%) "
            f"of samples assigned negative due to similarity close to random in-batch examples"
        )

    if return_stat:
        return df, ppl_filter_perc, sim_filter_perc
    return df


def label_with_semantic_preservation(df: pd.DataFrame, operation: Operation, pessimistic: bool = True) -> pd.DataFrame:
    """
    Label augmentation based on whether operation is meaning preserving.
    If pessimistic, if operation is labeled semantic preserving, we use it as positive.
        Otherwise, we label negative.
    If not pessimistic, we label only semantic changing ones negative, and all other positive
    """
    if pessimistic:
        df["aug_labels"] = "highly-meaning-preserving" in operation.keywords
    else:
        df["aug_labels"] = "meaning-alteration" not in operation.keywords
    return df


def generate_augmentation_label(
    df: pd.DataFrame,
    operation: Operation,
    method: Optional[str] = None,
    **kwargs,
) -> pd.DataFrame:
    """
    Given original and augmented sample, perplexity, similarity, type of operation, append
    augmentation label: whether the pair should be used as positive or negative augmentation
    """
    if method is None:
        method = "pessimistic"

    if method == "pessimistic":
        df = label_with_semantic_preservation(df, operation, pessimistic=True)
    elif method == "optimistic":
        df = label_with_semantic_preservation(df, operation, pessimistic=False)
    elif method == "lm_uniform":
        df = label_with_semantic_preservation_and_lm_uniform_threshold(df, operation, **kwargs)
    else:
        raise NotImplementedError(f"{method} method for generationg augmentation label is not implemented")

    df["aug_labels"] = df["aug_labels"].astype(int)
    return df


def transform_data_only(operation, dataset_name: str, split: str, batch_size: Optional[int]):
    if operation.name() == "NoTransform":
        dataset, pt_dataset = transform_existing_dataset(operation, dataset_name)
    else:
        split = process_split(dataset_name, split)
        dataset = _process_data(dataset_name, split, max_size=None if "train" in split else 1000)
        pt_dataset = dataset.apply_transformation(operation, batch_size=batch_size)
    return dataset, pt_dataset


def transform_existing_dataset(operation: Operation, dataset_name: str):
    if "nli_for_simcse" in dataset_name:
        df = pd.read_csv(dataset_name)
        dataset = UnsupervisedTextDataset(df.sent0)
        pt_dataset = UnsupervisedTextDataset(df.hard_neg)
    else:
        raise Exception(f"Dataset {dataset_name} with transformation {operation} not found")
    return dataset, pt_dataset


def process_split(dataset_name, split):
    percent = f"[{split.split('[')[-1]}" if "[" in split else ""
    if dataset_name == "mnli" and "train" not in split:
        split = f"validation_matched{percent}"
    elif dataset_name != "imdb" and "train" not in split:
        split = f"validation{percent}"
    return split


def create_baseline(dataset_name: str, split: str = "train[:100%]") -> str:
    normalized_dataset_name = normalize_dataset_names(dataset_name)
    final_dataset_path = f"{project_root}/dump/Baseline/{normalized_dataset_name}_{split[:-2].replace('[:', '_')}.csv"
    if os.path.isfile(final_dataset_path):
        print("Baseline dataset exist, skipping generation step ...")
    else:
        df = pd.DataFrame()
        os.makedirs(pathlib.Path(final_dataset_path).parent, exist_ok=True)
        split = process_split(dataset_name, split)
        dataset = _process_data(dataset_name, split, max_size=None if "train" in split else 1000)
        df["sent1"] = [raw_text for raw_text in dataset]
        df["aug_labels"] = 1
        df.to_csv(final_dataset_path, index=False)
    return final_dataset_path


def generate(operations: List[Operation], dataset_name: str, split: str, data_args) -> str:

    df = pd.DataFrame()
    if len(operations) == 1:
        # check if dataset exist in /dump/{augmentation}/{dataset}/, if so load it
        operation_hash = operations[0].name()
    else:
        operation_hash = f"comb_{len(operations)}_{hash_str('_'.join(sorted([o.name() for o in operations])))}"
    normalized_dataset_name = normalize_dataset_names(dataset_name)
    augmentation_arg_postfix = (
        ""
        if data_args.augmentation_init_args is None
        else "".join(f"{k}={v}" for k, v in json.loads(data_args.augmentation_init_args).items())
    )
    final_dataset_path = (
        f"{project_root}/dump/{operation_hash}/{normalized_dataset_name}_"
        f"{split[:-2].replace('[:', '_')}{augmentation_arg_postfix}.csv"
    )
    if os.path.isfile(final_dataset_path) and len(operations) > 1:
        print("Multi-Augmented dataset exist, skipping generation step ...")
        return final_dataset_path
    # else:
    os.makedirs(pathlib.Path(final_dataset_path).parent, exist_ok=True)
    for operation in operations:
        op_dataset_path = (
            f"{project_root}/dump/{operation.name()}/{normalized_dataset_name}_"
            f"{split[:-2].replace('[:', '_')}{augmentation_arg_postfix}.csv"
        )
        os.makedirs(pathlib.Path(op_dataset_path).parent, exist_ok=True)

        regenerate = True
        if os.path.isfile(op_dataset_path) and not data_args.force_regenerate:
            op_df = pd.read_csv(op_dataset_path)
            if "aug_labels" in op_df.columns:  # recompute augmentation labels everytime
                op_df = op_df.drop(columns=["aug_labels"])
            if (
                data_args.augmentation_label_method not in ["pessimistic", "optimistic"]
                and not ("highly-meaning-preserving" in operation.keywords)
                and not ("meaning-alteration" in operation.keywords)
                and "orig_ppl_sample" not in op_df.columns
            ):
                regenerate = True
            else:
                regenerate = False

        if regenerate:
            op_df = pd.DataFrame()
            if (
                data_args.augmentation_label_method in ["pessimistic", "optimistic"]
                or ("highly-meaning-preserving" in operation.keywords)
                or ("meaning-alteration" in operation.keywords)
            ):
                dataset, pt_dataset = transform_data_only(
                    operation, dataset_name, split, data_args.augmentation_batch_size
                )
            else:
                (_, similarity_samples, _, pt_dataset,) = evaluate_similarity.evaluate(
                    operation,
                    "sentence-transformers/all-mpnet-base-v2",
                    dataset_name,
                    split=split,
                    batch_size=64,
                    pt_dataset_path=op_dataset_path if os.path.isfile(op_dataset_path) else None,
                    return_dataset=True,
                    augmentation_batch_size=data_args.augmentation_batch_size,
                )
                pt_dataset.to_csv(op_dataset_path)
                op_df = pd.DataFrame(similarity_samples)
                (_, perplexity_samples, dataset, pt_dataset,) = evaluate_perplexity.evaluate(
                    operation,
                    "distilgpt2",
                    dataset_name,
                    split=split,
                    batch_size=16,
                    pt_dataset_path=op_dataset_path,
                    return_dataset=True,
                    augmentation_batch_size=data_args.augmentation_batch_size,
                )
                op_df = op_df.join(pd.DataFrame(perplexity_samples))

            if isinstance(pt_dataset, UnsupervisedTextDataset):
                op_df["sent1"] = [raw_text for raw_text in dataset]
                op_df["sent1_aug"] = [raw_text for raw_text in pt_dataset]
            else:
                op_df["sent1"] = [_get_instance_by_keys(list(raw_text)[:-1]) for raw_text in dataset]
                op_df["sent1_aug"] = [_get_instance_by_keys(list(raw_text)[:-1]) for raw_text in pt_dataset]

                if len(dataset.fields) > 2:  # KeyValueDataset
                    op_df["sent2"] = [list(raw_text)[1] for raw_text in dataset]
                    op_df["labels"] = [d["label"] for d in dataset.data]
                elif isinstance(pt_dataset, TextLineDataset):  # TextLineDataset
                    op_df["labels"] = dataset.labels
        op_df = generate_augmentation_label(
            op_df,
            operation,
            data_args.augmentation_label_method,
            **{
                "perplexity_std_scalar": data_args.augmentation_label_ppl_scalar,
                "similarity_std_scalar": data_args.augmentation_label_sim_scalar,
            },
        )
        op_df.to_csv(op_dataset_path, index=False)
        df = df.append(op_df)

    df.to_csv(final_dataset_path, index=False)
    print(f"Final aggregated dataset length: {len(df)}")
    return final_dataset_path
