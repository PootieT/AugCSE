import json
import logging
import os
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional
import torch


import transformers
from transformers import (
    MODEL_FOR_MASKED_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from transformers.file_utils import (
    cached_property,
    torch_required,
    is_torch_tpu_available,
)

from TestRunner import get_implementation
from tasks.TaskTypes import TaskType
from training import train_engine


logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# Adapted from https://github.com/princeton-nlp/SimCSE/blob/main/train.py


def is_json(myjson):
    try:
        json.loads(myjson)
    except ValueError as e:
        return False
    return True


@dataclass
class BasicArguments:
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
        return d


@dataclass
class ModelArguments(BasicArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    # Huggingface's original arguments
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"},
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

    # Our arguments
    pooler_type: str = field(
        default="avg",
        metadata={"help": "What kind of pooler to use (cls, avg, avg_top2, avg_first_last)."},
    )

    # projection layer related arguments
    projection_layers: int = field(
        default=0,
        metadata={
            "help": "How many layers of MLP after the pooler to use. Full projection layer is used for "
            "contrastive training. During evaluation, second to last layer projection is used as"
            "embedding for the sentence. if projection_layers = 100, it uses 2 layer batch norm MLP used in the paper."
        },
    )
    projection_heads: int = field(
        default=0,
        metadata={"help": "Number of projection heads to max pool over."},
    )
    moe: bool = field(
        default=False,
        metadata={"help": "If using Mixture of Expert for multi-head projection layer or not"},
    )
    moe_k: int = field(
        default=1,
        metadata={"help": "top k heads to select and pool over projections heads with"},
    )

    # loss related arugments
    supervised_augmentation_loss: bool = field(
        default=True,
        metadata={"help": "If using augmented example as additional supervised loss"},
    )

    contrastive_loss_type: Optional[str] = field(
        default="ranking",
        metadata={
            "help": "What kind of contrastive loss to use (None, ranking, ranking_pos_neg, "
            "triplet, alignment, contrastive, hard_contrastive)."
        },
    )
    temp: float = field(default=0.05, metadata={"help": "Temperature for cosine similarity."})
    margin: float = field(
        default=0.1,
        metadata={
            "help": "Triplet loss margin for contrastive learning. See:"
            "https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/losses/TripletLoss.py"
        },
    )
    alignment_alpha: int = field(
        default=2,
        metadata={"help": "Alignment alpha exponential coefficient. See: https://ssnl.github.io/hypersphere/"},
    )
    distance_metric: str = field(
        default="COSINE",
        metadata={"help": "What distance metric to use (EUCLIDEAN, MANHATTAN, COSINE)"},
    )
    uniform_t: int = field(
        default=2,
        metadata={"help": "T value in uniformity loss. See: https://ssnl.github.io/hypersphere/"},
    )
    hard_negative_weight: float = field(
        default=0,
        metadata={
            "help": "The **logit** of weight for hard negatives (only effective if hard negatives are used). "
            "(from SimCSE)"
        },
    )
    do_mlm: bool = field(default=False, metadata={"help": "Whether to use MLM auxiliary objective."})
    mlm_weight: float = field(
        default=0.1, metadata={"help": "Weight for MLM auxiliary objective (only effective if --do_mlm)."}
    )

    # discriminator related arguments
    discriminate: bool = field(
        default=False, metadata={"help": "Whether to add discriminator to predict augmentation type."}
    )
    gradient_reverse_multiplier: float = field(
        default=1.0,
        metadata={
            "help": "Multiplier for at gradient reversal layer. If set to negative number, gradient "
            "will not be reversed (only effective if --discriminate)."
        },
    )
    discriminator_layers: int = field(
        default=2,
        metadata={
            "help": "number of hidden layers in discriminator MLP"
        },
    )
    discriminator_dropout: float = field(
        default=0.2,
        metadata={"help": "dropout probability before classification"},
    )
    discriminate_original: bool = field(
        default=False,
        metadata={"help": "Whether to predict only whether augmentation is original sentence or augmented sentence"},
    )
    discriminate_order: bool = field(
        default=False, metadata={"help": "Whether to randomly shuffle aug vs original and predict the order"}
    )
    discriminator_weight: float = field(default=1.0, metadata={"help": "Weight for discriminator."})

    diff_cse_mode: bool = field(
        default=False,
        metadata={
            "help": "Use original sentence as positive for contrastive loss, but augmentations " "for discriminator"
        },
    )


@dataclass
class DataTrainingArguments(BasicArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # Huggingface's original arguments.
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."},
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"},
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    # Our arguments
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )

    train_file: Optional[str] = field(default=None, metadata={"help": "The training data file (.txt or .csv)."})
    max_seq_length: Optional[int] = field(
        default=32,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )

    # augmentation related arguments
    neutral_augmentations: str = field(
        default="",
        metadata={
            "help": "List of neutral augmentations to run comma separated. This dominates operation keywords."
                    "Neutral augmentation is used when user is unsure whether the augmentation should be "
                    "considered positive or negative augmentation. It is then automatically labeled given an"
                    "augmentation_label_method."
        },
    )
    positive_augmentations: str = field(
        default="",
        metadata={"help": "List of positive augmentations to run comma separated. This dominates operation keywords"},
    )
    negative_augmentations: str = field(
        default="",
        metadata={
            "help": "List of negative augmentations to run comma separated. This dominates operation keywords. To use"
                    "negative augmentation, the contrastive_loss_type needs to be one that uses negatives "
                    "(ex: ranking_pos_neg)"
        },
    )
    augmentation_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Batch size for augmentations that support batch transformation "
                          "(during augmentation generation)"},
    )
    augmentation_init_args: Optional[str] = field(
        default=None,
        metadata={"help": "JSON string specifying kwargs of parameters you want to pass into initiating augmentations"},
    )
    augmentation_label_method: str = field(
        default="pessimistic",
        metadata={
            "help": "Method to use when determining if augmented datapoint is positive or negative augmentation."
            "One of (pessimistic, optimistic, lm_uniform). This overrides augmentation semantic preservation labels."
                "Pessimistic: treats all augmentations as negative augmentations"
                "Optimistic: treats all augmentation as positive augmentations"
                "lm_uniform: uses perplexity difference or similarity between augmented sentence / original sentence as"
                    "labels for positivity. (ppl and sim needs to be pre-computed"
        },
    )
    augmentation_label_ppl_scalar: float = field(
        default=1.0,
        metadata={
            "help": "Scaling factor multiplied to perplexity standard deviation to determine negative augmentations."
            "The larger this value the more positive labels"
        },
    )
    augmentation_label_sim_scalar: float = field(
        default=0.2,
        metadata={
            "help": "Scaling factor multiplied to similarity standard deviation to determine negative augmentations."
            "The larger this value the more negative labels"
        },
    )
    ablate_positive_augmentations: bool = field(
        default=False,
        metadata={
            "help": "Whether to not select any augmentations as positives (test effect of negative augmentation)"
        },
    )
    ablate_negative_augmentations: bool = field(
        default=False,
        metadata={
            "help": "Whether to not select any augmentations as negatives (test effect of positive augmentation)"
        },
    )
    regenerate_augmentation_per_epoch: bool = field(
        default=False,
        metadata={"help": "Trigger new augmentation generation at the end of each epoch training"},
    )
    resample_augmentation_per_epoch: bool = field(
        default=False,
        metadata={"help": "Trigger augmentation resampling at the end of each epoch training"},
    )
    sample_default_augmentations: bool = field(
        default=False,
        metadata={"help": "Whether to keep 1/#augmentations of all dataset as default augmentations"},
    )
    task_type: TaskType = field(
        default=TaskType.TEXT_CLASSIFICATION,
        metadata={"help": "Task type for the augmentation"},
    )
    force_regenerate: bool = field(
        default=False,
        metadata={"help": "Whether to regenerate all the augmented data (and ignore cache)"},
    )
    uniform_augmentation_sampling: bool = field(
        default=False,
        metadata={
            "help": "Whether to randomly sample one of augmentations for each data point at beginning of training"
        },
    )
    remove_no_augmentations: bool = field(
        default=False,
        metadata={
            "help": "Whether to remove datapoints with no perturbation from the augmentation methods. This is good at"
            "removing noise when negative augmentation fails and produce same sentence as original."
        },
    )  # TODO may need other methods for non-sentence embedding models (classification, ex)

    def __post_init__(self):
        if self.augmentation_init_args is not None:
            assert is_json(self.augmentation_init_args), "kwargs for augmentation needs to be json parsable"
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv",
                    "json",
                    "txt",
                ], "`train_file` should be a csv, a json or a txt file."


@dataclass
class OurTrainingArguments(TrainingArguments):
    # Evaluation
    ## By default, we evaluate STS (dev) during training (for selecting best checkpoints) and evaluate
    ## both STS and transfer tasks (dev) at the end of training. Using --eval_transfer will allow evaluating
    ## both STS and transfer tasks (dev) during training.
    eval_original: bool = field(
        default=True,
        metadata={"help": "Evaluate original validation set of the dataset."},
    )
    eval_transfer: bool = field(
        default=False,
        metadata={"help": "Evaluate transfer task dev sets (in validation)."},
    )
    eval_robust: bool = field(
        default=False,
        metadata={"help": "Evaluate augmentation robustness tasks "},
    )
    eval_glue: bool = field(
        default=False,
        metadata={"help": "Evaluate glue tasks "},
    )
    wandb_project: Optional[str] = field(
        default=None,
        metadata={"help": "Weights and Biases project name. Default is huggingface"},
    )
    hyper_path_modifier: Optional[str] = field(
        default=None,
        metadata={
            "help": "Which hyperparameter to use to create unique output path. For example:"
            "'default-hard_negative_weight' will expand to 'default-HNW=-5`"
            "in actual output path"
        },
    )

    @cached_property
    @torch_required
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            self._n_gpu = 0
        elif is_torch_tpu_available():
            device = xm.xla_device()
            self._n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
            # the default value.
            self._n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
            #
            # deepspeed performs its own DDP internally, and requires the program to be started with:
            # deepspeed  ./program.py
            # rather than:
            # python -m torch.distributed.launch --nproc_per_node=2 ./program.py
            if self.deepspeed:
                from .integrations import is_deepspeed_available

                if not is_deepspeed_available():
                    raise ImportError("--deepspeed requires deepspeed: `pip install deepspeed`.")
                import deepspeed

                deepspeed.init_distributed()
            else:
                torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            self._n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device


def parse_args():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, OurTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        (
            model_args,
            data_args,
            training_args,
        ) = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def main(model_args, data_args, training_args):
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    os.environ["WANDB_PROJECT"] = training_args.wandb_project

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Identify the transformation that the user has mentioned and override meaning preservation keywords
    positive_augmentations = (
        [get_implementation(aug) for aug in data_args.positive_augmentations.split(",")]
        if data_args.positive_augmentations
        else []
    )
    for aug in positive_augmentations:
        aug.keywords = [] if aug.keywords is None else aug.keywords
        aug.keywords.append("highly-meaning-preserving")
        if "meaning-alteration" in aug.keywords:
            aug.keywords.remove("meaning-alteration")
    negative_augmentations = (
        [get_implementation(aug) for aug in data_args.negative_augmentations.split(",")]
        if data_args.negative_augmentations
        else []
    )
    for aug in negative_augmentations:
        aug.keywords = [] if aug.keywords is None else aug.keywords
        aug.keywords.append("meaning-alteration")
        if "highly-meaning-preserving" in aug.keywords:
            aug.keywords.remove("highly-meaning-preserving")
    neutral_augmentations = (
        [get_implementation(aug) for aug in data_args.neutral_augmentations.split(",")]
        if data_args.neutral_augmentations
        else []
    )
    for aug in neutral_augmentations:
        aug.keywords = [] if aug.keywords is None else aug.keywords
        if "highly-meaning-preserving" in aug.keywords:
            aug.keywords.remove("highly-meaning-preserving")
        if "meaning-alteration" in aug.keywords:
            aug.keywords.remove("meaning-alteration")
    data_args.augmentations = [*positive_augmentations, *negative_augmentations, *neutral_augmentations]

    if (
        len(data_args.augmentations) > 0
        and not data_args.uniform_augmentation_sampling
        and model_args.contrastive_loss_type.startswith("ranking_")
    ):
        training_args.use_pos_neg_pipeline = True
    else:
        training_args.use_pos_neg_pipeline = False
    print(f"Using pos_neg_pipeline={training_args.use_pos_neg_pipeline}")
    data_args.cache_dir = model_args.cache_dir
    data_args.contrastive_loss_type = model_args.contrastive_loss_type
    data_args.discriminate_original = model_args.discriminate_original
    data_args.discriminate_order = model_args.discriminate_order

    train_engine.train(
        implementations=data_args.augmentations,
        task_type=data_args.task_type,
        model_args=model_args,
        data_args=data_args,
        training_args=training_args,
        logger=logger,
    )


if __name__ == "__main__":
    model_args, data_args, training_args = parse_args()
    main(model_args, data_args, training_args)
