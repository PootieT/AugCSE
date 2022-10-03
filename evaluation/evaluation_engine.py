import os
import json
import pickle
from typing import Dict, Any, List, Tuple, Union, Optional
import pathlib

from evaluation import (
    evaluate_ner_tagging,
    evaluate_paraphrase_detection,
    evaluate_question_answering,
    evaluate_text_classification,
    evaluate_text_generation,
    evaluate_similarity,
    evaluate_perplexity,
)
from interfaces.QuestionAnswerOperation import QuestionAnswerOperation
from interfaces.SentenceOperation import SentenceOperation
from interfaces.SentencePairOperation import SentencePairOperation
from interfaces.TaggingOperation import TaggingOperation
from tasks.TaskTypes import TaskType

"""
This is the evaluation engine.
Currently has been implemented for SentenceTransformation:
eg. python evaluate.py -t ButterFingersPerturbation
"""


def evaluate(
    implementation,
    task_type: str,
    language="en",
    model=None,
    dataset=None,
    percentage_of_examples=None,
    evaluate_filter=False,
    dump_results=True,
    batch_size=8,
):
    # The evaluation engine would effectively do the following
    # (1) Loading a standard model and a test set (the model's original test set would be the best choice)
    # (2) Executing perturbations to generate the perturbed test set.
    # (3) Executing these against the model and evaluate its performance (display nicely :P )
    # (4) Writing a neat README.
    task_type = get_task_type(implementation, task_type)
    results = execute_model(
        implementation,
        evaluate_filter=evaluate_filter,
        task_type=task_type,
        locale=language,
        model_name=model,
        dataset=dataset,
        percentage_of_examples=percentage_of_examples,
        batch_size=batch_size,
    )
    if dump_results:
        if isinstance(results, dict):
            dump_results_to_dir(implementation.name(), task_type, results)
        else:  # if not just singular performance report, separate out np.arrays that needs to be pickled
            dump_results_to_dir(
                implementation.name(), task_type, results[0], results[1]
            )
    return


def evaluate_mt(
    implementation,
    task_type,
    src_locale="en",
    tgt_locale="en",
    model=None,
    dataset=None,
    percent_of_examples=None,
    evaluate_filter=False,
):
    # TODO
    return


def get_task_type(implementation, task_type):
    if task_type is None:
        print(
            "Undefined task type, switching to default task %s",
            implementation.tasks[0].name,
        )
        return str(implementation.tasks[0]).split(".")[1]
    return task_type


def execute_model(
    implementation,
    task_type,
    locale="en",
    model_name=None,
    dataset=None,
    percentage_of_examples=20,
    evaluate_filter=False,
    batch_size=8,
) -> Union[Tuple[Dict[str, float], Dict[str, Any]], Dict[str, Any]]:
    interface = implementation.__bases__[0]  # SentenceTransformation
    impl = implementation()
    if locale in ["en", "zh"]:
        if (
            isinstance(impl, SentenceOperation)
            and TaskType[task_type] == TaskType.TEXT_CLASSIFICATION
        ):
            return evaluate_text_classification.evaluate(
                impl,
                evaluate_filter,
                model_name,
                dataset,
                split=f"test[:{percentage_of_examples}%]",
            )

        elif (
            isinstance(impl, QuestionAnswerOperation)
            and TaskType[task_type] == TaskType.QUESTION_ANSWERING
        ):
            return evaluate_question_answering.evaluate(
                impl,
                evaluate_filter,
                model_name,
                dataset,
                split=f"validation[:{percentage_of_examples}%]",
            )

        elif (
            isinstance(impl, SentenceOperation)
            and TaskType[task_type] == TaskType.TEXT_TO_TEXT_GENERATION
        ):
            return evaluate_text_generation.evaluate(
                impl,
                evaluate_filter,
                model_name,
                dataset,
                split=f"test[:{percentage_of_examples}%]",
            )

        elif (
            isinstance(impl, TaggingOperation)
            and TaskType[task_type] == TaskType.TEXT_TAGGING
        ):
            return evaluate_ner_tagging.evaluate(
                impl,
                evaluate_filter,
                model_name,
                dataset,
                split=f"test[:{percentage_of_examples}%]",
            )

        elif (
            isinstance(impl, SentencePairOperation)
            and TaskType[task_type] == TaskType.PARAPHRASE_DETECTION
        ):
            return evaluate_paraphrase_detection.evaluate(
                impl,
                evaluate_filter,
                model_name,
                dataset,
                split=f"test[:{percentage_of_examples}%]",
            )
        # Other if else cases should be added here.
        if (
            isinstance(impl, SentenceOperation)
            and TaskType[task_type] == TaskType.SIMILARITY_EXP
        ):
            metric, feature = evaluate_similarity.evaluate(
                impl,
                model_name,
                dataset,
                split=f"test[:{percentage_of_examples}%]",
                batch_size=batch_size,
            )
            return metric, feature
        if (
            isinstance(impl, SentenceOperation)
            and TaskType[task_type] == TaskType.PERPLEXITY_EXP
        ):
            metric, feature = evaluate_perplexity.evaluate(
                impl,
                model_name,
                dataset,
                split=f"test[:{percentage_of_examples}%]",
                batch_size=batch_size,
            )
            return metric, feature
        else:
            print(
                f"No default evaluation model exists for the interface {interface} in the locale {locale}."
                f"It's okay to skip the evaluation for the purpose of the PR. If you are interested to evaluate "
                f"your perturbation on a task and a dataset, "
                f"the right place to do it would to add a new class in the evaluation folder "
                f"and call it from execute_model. That's it!"
            )
    else:
        print(
            f"No default evaluation model exists in the locale {locale}."
            f"It's okay to skip the evaluation for the purpose of the PR. If you are interested to evaluate "
            f"your perturbation on a task and a dataset, "
            f"the right place to do it would to add a new class in the evaluation folder "
            f"and call it from execute_model. That's it!"
        )


def dump_results_to_dir(
    augmentation_name: str,
    task_type: str,
    metrics: Dict[str, Union[float, str]],
    features: Optional[Dict[str, Any]] = None,
):
    project_root = pathlib.Path(__file__).parent.resolve().parent
    model_str = metrics["model_name"].replace("/", "_")
    dump_path = f"{project_root}/dump/{augmentation_name}/{model_str}"
    if not os.path.isdir(dump_path):
        os.makedirs(dump_path)

    file_str = f"{task_type.lower()}_{metrics['dataset_name']}_{metrics['split'].replace('[:','_').replace('%]','')}"
    json.dump(metrics, open(f"{dump_path}/{file_str}.json", "w"))
    pickle.dump(features, open(f"{dump_path}/{file_str}.p", "wb"))
