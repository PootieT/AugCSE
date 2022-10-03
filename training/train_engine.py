import os
import pathlib
from typing import List

from evaluation.evaluation_engine import get_task_type
from interfaces.Operation import Operation
from interfaces.QuestionAnswerOperation import QuestionAnswerOperation
from interfaces.SentenceOperation import SentenceOperation
from interfaces.SentencePairOperation import SentencePairOperation
from interfaces.TaggingOperation import TaggingOperation
from tasks.TaskTypes import TaskType
from training import generate_text_classification, train_text_classification
from training.generate_text_classification import create_baseline
from training.utils import normalize_dataset_names, expand_hyper_identifier_str, initialize_implementations


def set_run_metadata(data_args, model_args, train_path, training_args):
    training_args.output_dir = str(
        pathlib.Path(train_path)
        .parent.joinpath(model_args.model_name_or_path.replace("/", "_"))
        .joinpath(normalize_dataset_names(data_args.dataset_name))
    )
    if training_args.hyper_path_modifier is not None:
        extended_path = expand_hyper_identifier_str(
            [data_args, training_args, model_args], training_args.hyper_path_modifier
        )
        training_args.output_dir = str(pathlib.Path(training_args.output_dir).joinpath(extended_path))

    if os.path.exists(training_args.output_dir) and not training_args.overwrite_output_dir:
        raise Exception(f"training output directory {training_args.output_dir} exist and overwrite is not allowed")
    else:
        os.makedirs(training_args.output_dir, exist_ok=True)

    training_args.logging_dir = str(pathlib.Path(training_args.output_dir).joinpath("log"))
    path_split = training_args.output_dir.split("/")
    training_args.run_name = (
        f"{path_split[-4][:10]}_{path_split[-3][:10]}_{path_split[-2][:10]}_{extended_path}"
        if training_args.hyper_path_modifier
        else f"{path_split[-3][:10]}_{path_split[-2][:10]}_{path_split[-1][:10]}"
    )


def train(implementations: List[Operation], task_type: str, model_args, data_args, training_args, logger):
    # The train engine would effectively do the following
    # (1) Loading a standard model and a test set (the model's original test set would be the best choice)
    # (2) Executing perturbations to generate the perturbed test set.
    # (3) Executing these against the model and evaluate its performance (display nicely :P )
    # (4) Writing a neat README.
    if len(implementations) > 0:
        task_type = get_task_type(implementations[0], task_type)
        assert all(
            (task_type in impl.tasks) for impl in implementations
        ), f"Not all transformations are for task {task_type}"
    else:  # unsupervised, simcse baseline
        task_type = TaskType.TEXT_CLASSIFICATION
    results = execute_model(implementations, task_type, "en", model_args, data_args, training_args, logger)
    return


def execute_model(implementations, task_type, locale, model_args, data_args, training_args, logger):
    if len(implementations) > 0:
        interface = implementations[0].__bases__[0]  # SentenceTransformation
    else:
        interface = SentenceOperation

    impls = initialize_implementations(implementations, data_args.augmentation_init_args)
    if locale in ["en", "zh"]:
        if task_type == TaskType.TEXT_CLASSIFICATION:
            if len(impls) > 0 and isinstance(impls[0], SentenceOperation):
                train_path = generate_text_classification.generate(
                    impls, data_args.dataset_name, "train[:100%]", data_args
                )
            else:  # no augmentation
                train_path = create_baseline(data_args.dataset_name, "train[:100%]")

            data_args.train_file = train_path
            set_run_metadata(data_args, model_args, train_path, training_args)
            return train_text_classification.train(model_args, data_args, training_args, logger)

        elif isinstance(impls[0], QuestionAnswerOperation) and TaskType[task_type] == TaskType.QUESTION_ANSWERING:
            raise NotImplementedError("QA training not implemented")

        elif isinstance(impls[0], SentenceOperation) and TaskType[task_type] == TaskType.TEXT_TO_TEXT_GENERATION:
            raise NotImplementedError("Text generation training not implemented")

        elif isinstance(impls[0], TaggingOperation) and TaskType[task_type] == TaskType.TEXT_TAGGING:
            raise NotImplementedError("NER tagging training not implemented")

        elif isinstance(impls[0], SentencePairOperation) and TaskType[task_type] == TaskType.PARAPHRASE_DETECTION:
            raise NotImplementedError("Paraphrase detection training not implemented")
        # Other if else cases should be added here.
        else:
            print(
                f"No default training model exists for the interface {interface} in the locale {locale}."
                f"It's okay to skip the training for the purpose of the PR. If you are interested to training "
                f"your perturbation on a task and a dataset, "
                f"the right place to do it would to add a new class in the training folder "
                f"and call it from execute_model. That's it!"
            )
    else:
        print(
            f"No default training model exists in the locale {locale}."
            f"It's okay to skip the training for the purpose of the PR. If you are interested to training "
            f"your perturbation on a task and a dataset, "
            f"the right place to do it would to add a new class in the training folder "
            f"and call it from execute_model. That's it!"
        )
