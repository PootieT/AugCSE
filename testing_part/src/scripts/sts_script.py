import argparse
from testing_part.src.experimenters.sts_experimenter import SentExperimenter
from testing_part.src.test_datasets.nli_dataset import NLIDataset
from testing_part.src.test_datasets.cls_dataset import CLSDataset

parser = argparse.ArgumentParser(
    description="This is the custom, cross task and cross dataset evaluation script for the Data Agumentation project"
)

parser.add_argument("--task_type", "-task", help="type of the task")
parser.add_argument(
    "--model",
    "-m",
    help="HuggingFace model to evaluate. Note that the model should be in HF-models.",
)
parser.add_argument(
    "-b",
    "--batch_size",
    help="batch size for evaluation tasks",
    default=8,
)
args = parser.parse_args()

# These 4 steps run the experiments you can use them in your loops as well.
sts_experimenter = SentExperimenter()
model, tokenizer = sts_experimenter.load_model("princeton-nlp/unsup-simcse-roberta-base")
if args.task_type == "nli" or args.task_type == "all":
    nli_dataset = NLIDataset()
    sts_experimenter.mnli_experiment(model, tokenizer, nli_dataset)
if args.task_type == "cola" or args.task_type == "all":
    cls_train_dataset = CLSDataset(split_type="train")
    cls_val_dataset = CLSDataset()
    sts_experimenter.cola_experiment(model, tokenizer, cls_train_dataset, cls_val_dataset, eval_type="linear")
