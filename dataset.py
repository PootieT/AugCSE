from __future__ import annotations

from typing import Iterable, List, Optional

from datasets import Dataset
from tqdm import tqdm
import pandas as pd

from interfaces import Operation
from interfaces.QuestionAnswerOperation import QuestionAnswerOperation
from interfaces.SentenceOperation import (
    SentenceAndTargetOperation,
    SentenceAndTargetsOperation,
    SentenceOperation,
)
from interfaces.SentencePairOperation import SentencePairOperation
from tasks.TaskTypes import TaskType


class BaseDataset(Iterable):
    def __init__(self, data: Iterable):
        self.data = data

    def apply_filter(self, condition: Operation):
        raise NotImplementedError("BaseDataset does not implement this function.")

    def apply_transformation(self, transformation: Operation, batch_size: Optional[int] = None):
        raise NotImplementedError("BaseDataset does not implement this function.")

    def to_csv(self, path: str):
        raise NotImplementedError("BaseDataset does not implement this function.")

    def __iter__(self):
        raise NotImplementedError("BaseDataset does not implement this function.")

    def __len__(self):
        raise NotImplementedError("BaseDataset does not implement this function.")

    def __or__(self, other):
        raise NotImplementedError("BaseDataset does not implement this function.")

    def __and__(self, other):
        raise NotImplementedError("BaseDataset does not implement this function.")

    def __sub__(self, other):
        raise NotImplementedError("BaseDataset does not implement this function")


"""
Dataset for data in plain texts where each line is a datapoint (unsupervised)
"""


class UnsupervisedTextDataset(BaseDataset):
    tasks = [TaskType.TEXT_CLASSIFICATION]

    def __init__(self, data: List[str], perturbation_rate: float = -1.0):
        super(UnsupervisedTextDataset, self).__init__(data)
        self.perturbation_rate = perturbation_rate

    @classmethod
    def from_huggingface(cls, dataset: Dataset, task_type, fields, max_size=None):
        data = []
        max_size = max_size or len(dataset)
        if max_size < len(dataset):
            dataset = dataset.select(range(max_size))

        for example in dataset:
            data.append(example[fields[0]])
        return cls(data)

    def apply_filter(self, filter: SentenceOperation) -> UnsupervisedTextDataset:
        filtered_data = []
        print("Applying filtering:")
        for datapoint in tqdm(self.data, total=len(self.data)):
            if filter.filter(datapoint):
                filtered_data.append(datapoint)

        return UnsupervisedTextDataset(filtered_data)

    def apply_transformation(
        self, transformation: SentenceOperation, batch_size: Optional[int] = None
    ) -> UnsupervisedTextDataset:
        transformed_data = []
        print("Applying transformation:")

        # calculating ratio of transformed example to unchanged example
        successful_num = 0
        failed_num = 0

        if batch_size is not None:
            assert hasattr(transformation, "generate_batch"), "transformation needs a generate_batch method"
            for idx in tqdm(range(0, len(self.data), batch_size)):
                datapoints = self.data[idx : idx + batch_size]
                pt_examples_batch = transformation.generate_batch(datapoints)
                for j, datapoint in enumerate(datapoints):
                    pt_examples = pt_examples_batch[j]
                    successful_pt, failed_pt = transformation.compare(datapoint, pt_examples)
                    successful_num += successful_pt
                    failed_num += failed_pt

                    if len(pt_examples) == 0:
                        pt_examples = [datapoint]  # if no transformation copy over original data
                    transformed_data.extend(pt_examples)
        else:
            for datapoint in tqdm(self.data, total=len(self.data)):
                pt_examples = transformation.generate(datapoint)
                successful_pt, failed_pt = transformation.compare(datapoint, pt_examples)
                successful_num += successful_pt
                failed_num += failed_pt

                if len(pt_examples) == 0:
                    pt_examples = [datapoint]  # if no transformation copy over original data
                transformed_data.extend(pt_examples)

        total_num = successful_num + failed_num
        pt_rate = successful_num / total_num if total_num > 0 else 0
        print(
            "Finished transformation! {} examples generated from {} original examples, with {} successfully transformed and {} unchanged ({} perturb rate)".format(
                total_num,
                len(self.data),
                successful_num,
                failed_num,
                pt_rate,
            )
        )
        if total_num == 0:
            return None
        return UnsupervisedTextDataset(transformed_data, pt_rate)

    def to_csv(self, path: str):
        df = pd.DataFrame({"text": self.data})
        df.to_csv(path, index=False)

    def __iter__(self):
        for text in self.data:
            yield text

    def __len__(self):
        return len(self.data)

    def __or__(self, other: UnsupervisedTextDataset) -> UnsupervisedTextDataset:
        data = list(set(self.data).union(set(other.data)))
        return UnsupervisedTextDataset(data)

    def __and__(self, other: UnsupervisedTextDataset) -> UnsupervisedTextDataset:
        data = list(set(self.data).intersection(set(other.data)))
        return UnsupervisedTextDataset(data)

    def __sub__(self, other: UnsupervisedTextDataset) -> UnsupervisedTextDataset:
        data = list(set(self.data).difference(set(other.data)))
        return UnsupervisedTextDataset(data)


"""
Dataset for data in plain texts where each line is a datapoint (with label)
"""


class TextLineDataset(BaseDataset):
    tasks = [TaskType.TEXT_CLASSIFICATION]

    def __init__(self, data: List[str], labels: List, perturbation_rate: float = -1.0):
        super(TextLineDataset, self).__init__(data)
        assert len(data) == len(labels), "The number of datapoint should be the same as the number of labels"
        self.labels = labels
        self.mapping = {datapoint: label for datapoint, label in zip(self.data, self.labels)}
        self.perturbation_rate = perturbation_rate

    @classmethod
    def from_huggingface(cls, dataset, task_type, fields, max_size=None):
        data = []
        labels = []
        max_size = max_size or len(dataset)
        for example in list(dataset)[:max_size]:
            data.append(example[fields[0]])
            labels.append(example[fields[1]])
        return cls(data, labels)

    def apply_filter(self, filter: SentenceOperation) -> TextLineDataset:
        filtered_data = []
        filtered_labels = []
        print("Applying filtering:")
        for datapoint, label in tqdm(zip(self.data, self.labels), total=len(self.data)):
            if filter.filter(datapoint):
                filtered_data.append(datapoint)
                filtered_labels.append(label)

        return TextLineDataset(filtered_data, filtered_labels)

    def apply_transformation(
        self, transformation: SentenceOperation, batch_size: Optional[int] = None
    ) -> TextLineDataset:
        transformed_data = []
        transformed_labels = []
        print("Applying transformation:")

        # calculating ratio of transformed example to unchanged example
        successful_num = 0
        failed_num = 0

        for datapoint, label in tqdm(zip(self.data, self.labels), total=len(self.data)):
            pt_examples = transformation.generate(datapoint)
            successful_pt, failed_pt = transformation.compare(datapoint, pt_examples)
            successful_num += successful_pt
            failed_num += failed_pt

            if len(pt_examples) == 0:
                pt_examples = [datapoint]  # if no transformation copy over original data
            transformed_data.extend(pt_examples)
            transformed_labels.extend([label] * len(pt_examples))

        total_num = successful_num + failed_num
        pt_rate = successful_num / total_num if total_num > 0 else 0
        print(
            "Finished transformation! {} examples generated from {} original examples, with {} successfully transformed and {} unchanged ({} perturb rate)".format(
                total_num,
                len(self.data),
                successful_num,
                failed_num,
                pt_rate,
            )
        )
        if total_num == 0:
            return None
        return TextLineDataset(transformed_data, transformed_labels, pt_rate)

    def to_csv(self, path: str):
        df = pd.DataFrame({"text": self.data, "label": self.labels})
        df.to_csv(path, index=False)

    def __iter__(self):
        for text, label in zip(self.data, self.labels):
            yield (text, label)

    def __len__(self):
        return len(self.data)

    def __or__(self, other: TextLineDataset) -> TextLineDataset:
        data = list(set(self.data).union(set(other.data)))
        mapping = {**self.mapping, **other.mapping}
        labels = [mapping[datapoint] for datapoint in data]
        return TextLineDataset(data, labels)

    def __and__(self, other: TextLineDataset) -> TextLineDataset:
        data = list(set(self.data).intersection(set(other.data)))
        labels = [self.mapping[datapoint] for datapoint in data]
        return TextLineDataset(data, labels)

    def __sub__(self, other: TextLineDataset) -> TextLineDataset:
        data = list(set(self.data).difference(set(other.data)))
        labels = [self.mapping[datapoint] for datapoint in data]
        return TextLineDataset(data, labels)


"""
Dataset for data in format of key-value pairs, e.g. data read from jsonl file
"""


class KeyValueDataset(BaseDataset):
    tasks = [
        TaskType.TEXT_TO_TEXT_GENERATION,
        TaskType.QUESTION_ANSWERING,
        TaskType.QUESTION_GENERATION,
        TaskType.TEXT_CLASSIFICATION,  # for >1 field classification
    ]

    # data: input data samples read from jsonl file
    # task_type: task type specified
    # fields: list of relevant keys (e.g. to your sentence/target, context/question/answer, etc.)
    #         The number of keys should be aligned with the transform/filter operation.
    def __init__(
        self,
        data: List[dict],
        task_type=TaskType.TEXT_TO_TEXT_GENERATION,
        fields: List[str] = None,
        perturbation_rate: float = -1.0,
    ):
        super(KeyValueDataset, self).__init__(data)
        self.task_type = task_type
        self.fields = fields
        self.operation_type = None
        self.perturbation_rate = perturbation_rate

    @classmethod
    def from_huggingface(cls, dataset, task_type, fields, max_size=None):
        data = []
        max_size = max_size or len(dataset)
        if task_type not in [
            TaskType.QUESTION_ANSWERING,
            TaskType.QUESTION_GENERATION,
        ]:
            for example in list(dataset)[:max_size]:
                data.append({key: example[key] for key in fields})
        else:
            # this is an ugly implementation, which hard-codes the squad data format
            # TODO might need a more elegant way to deal with the fields with hierarchy, e.g. the answers field in squad data (exampl['answers']['text'])
            for example in list(dataset)[:max_size]:
                data.append(
                    {
                        fields[0]: example[fields[0]],
                        fields[1]: example[fields[1]],
                        fields[2]: example[fields[2]]["text"],
                    }
                )
        return cls(data, task_type, fields)

    def _analyze(self, subfields: List[str]):
        if subfields is None:
            subfields = self.fields

        assert set(subfields) <= set(self.fields), "Your can only choose from fields within {}".format(self.fields)

        if self.task_type == TaskType.TEXT_TO_TEXT_GENERATION:
            if len(subfields) == 1:
                self.operation_type = "sentence"
            elif len(subfields) == 2:
                self.operation_type = "sentence_and_target"
            elif len(subfields) > 2:
                self.operation_type = "sentence_and_targets"
        elif self.task_type in [
            TaskType.QUESTION_ANSWERING,
            TaskType.QUESTION_GENERATION,
        ]:
            # this is in case that one would like to use SentenceOperation (e.g. butter finger) on specific fields (e.g. only the question)
            if len(subfields) == 1:
                self.operation_type = "sentence"
            elif len(subfields) == 2:
                self.operation_type = "sentence_and_target"
            else:
                self.operation_type = "question_answer"
        elif self.task_type in [TaskType.TEXT_CLASSIFICATION]:
            self.operation_type = "sentence"
        elif self.task_type in [TaskType.PARAPHRASE_DETECTION]:
            self.operation_type = "sentence1_sentence2_target"

        filter_func = self.__getattribute__("_apply_" + self.operation_type + "_filter")
        transformation_func = self.__getattribute__("_apply_" + self.operation_type + "_transformation")
        return filter_func, transformation_func

    # this function is an adapter and will call the corresponding filter function for the task
    # subfields: the fields to apply filter, it is a subset of self.fields
    def apply_filter(self, filter: Operation, subfields: List[str] = None) -> KeyValueDataset:
        filter_func, _ = self._analyze(subfields)

        filtered_data = []
        print("Applying filtering:")
        for datapoint in tqdm(self.data):
            if filter_func(datapoint, filter):
                filtered_data.append(datapoint)

        return KeyValueDataset(filtered_data, self.task_type, self.fields)

    def _apply_sentence_filter(self, datapoint: dict, filter: SentenceOperation):
        sentence = datapoint[self.fields[0]]
        return filter.filter(sentence)

    def _apply_sentence_and_target_filter(self, datapoint: dict, filter: SentenceAndTargetOperation):
        sentence = datapoint[self.fields[0]]
        target = datapoint[self.fields[1]]
        return filter.filter(sentence, target)

    def _apply_sentence_and_targets_filter(self, datapoint: dict, filter: SentenceAndTargetsOperation):
        sentence = datapoint[self.fields[0]]
        targets = [datapoint[target_key] for target_key in self.fields[1:]]
        return filter.filter(sentence, targets)

    def _apply_question_answer_filter(self, datapoint: dict, filter: QuestionAnswerOperation):
        context = datapoint[self.fields[0]]
        question = datapoint[self.fields[1]]
        answers = [datapoint[answer_key] for answer_key in self.fields[2:]]
        return filter.filter(context, question, answers[0])  # @Zhenhao, converting answers to answers[0] here

    def _apply_sentence1_sentence2_target_filter(self, datapoint: dict, filter: SentencePairOperation):
        """Apply a filter to SentencePairOperations."""
        sentence1 = datapoint[self.fields[0]]
        sentence2 = datapoint[self.fields[1]]
        target = datapoint[self.fields[2]]
        return filter.filter(sentence1, sentence2, str(target))

    # this function is an adapter and will call the corresponding transform function for the task
    # subfields: the fields to apply transformation, it is a subset of self.fields
    def apply_transformation(
        self, transformation: Operation, subfields: List[str] = None, batch_size: Optional[int] = None
    ) -> KeyValueDataset:
        _, transformation_func = self._analyze(subfields)
        transformed_data = []
        print("Applying transformation:")

        # calculating ratio of transformed example to unchanged example
        successful_num = 0
        failed_num = 0

        for datapoint in tqdm(self.data):
            pt_examples = transformation_func(datapoint.copy(), transformation)
            successful_pt, failed_pt = transformation.compare(datapoint, pt_examples)
            successful_num += successful_pt
            failed_num += failed_pt

            transformed_data.extend(pt_examples)  # don't want self.data to be changed

        total_num = successful_num + failed_num
        pt_rate = successful_num / total_num if total_num > 0 else 0
        print(
            "Finished transformation! {} examples generated from {} original examples, with {} successfully transformed and {} unchanged ({} perturb rate)".format(
                total_num,
                len(self.data),
                successful_num,
                failed_num,
                pt_rate,
            )
        )
        if total_num == 0:
            return None
        return KeyValueDataset(transformed_data, self.task_type, self.fields, pt_rate)

    def _apply_sentence_transformation(self, datapoint: dict, transformation: SentenceOperation):
        sentence = datapoint[self.fields[0]]
        transformed_sentences = transformation.generate(sentence)

        if len(self.fields) > 1:  # QQP, MNLI
            pt_datapoints = []
            for tr in transformed_sentences:
                pt_datapoint = datapoint.copy()
                pt_datapoint[self.fields[0]] = tr
                pt_datapoints.append(pt_datapoint)
            return pt_datapoints

        datapoint[self.fields[0]] = transformed_sentences
        return [datapoint]

    def _apply_sentence_and_target_transformation(self, datapoint: dict, transformation: SentenceAndTargetOperation):
        sentence = datapoint[self.fields[0]]
        target = datapoint[self.fields[1]]
        transformed_sentence, transformed_target = transformation.generate(sentence, target)
        datapoint[self.fields[0]] = transformed_sentence
        datapoint[self.fields[1]] = transformed_target
        return [datapoint]

    def _apply_sentence_and_targets_transformation(self, datapoint: dict, transformation: SentenceAndTargetsOperation):
        sentence = datapoint[self.fields[0]]
        targets = [datapoint[target_key] for target_key in self.fields[1:]]
        transformed = transformation.generate(sentence, targets)
        datapoints = []
        for to in transformed:
            datapoint_n = dict()
            datapoint_n[self.fields[0]] = to[0]
            for i, target_key in enumerate(self.fields[1:]):
                datapoint[target_key] = to[1][1 + i]  # targets starting from pos 1
            datapoints.append(datapoint_n)
        return datapoints

    def _apply_question_answer_transformation(self, datapoint: dict, transformation: QuestionAnswerOperation):
        context = datapoint[self.fields[0]]
        question = datapoint[self.fields[1]]
        answers = [datapoint[answer_key] for answer_key in self.fields[2:]]
        transformed = transformation.generate(context, question, answers)

        datapoints = []
        for to in transformed:
            datapoint_n = dict()
            datapoint_n[self.fields[0]] = to[0]
            datapoint_n[self.fields[1]] = to[1]
            for i, answers_key in enumerate(self.fields[2:]):
                datapoint_n[answers_key] = to[2 + i]  # answers starting from pos 2
            datapoints.append(datapoint_n)

        return datapoints

    def _apply_sentence1_sentence2_target_transformation(self, datapoint: dict, transformation: SentencePairOperation):
        """Apply a transformation to SentencePairOperations."""
        sentence1 = datapoint[self.fields[0]]
        sentence2 = datapoint[self.fields[1]]
        target = datapoint[self.fields[2]]
        target_type = type(target)

        transformed = transformation.generate(sentence1, sentence2, str(target))
        datapoints = []
        for to in transformed:
            datapoint_n = dict()
            datapoint_n[self.fields[0]] = to[0]
            datapoint_n[self.fields[1]] = to[1]
            datapoint_n[self.fields[2]] = target_type(to[2])
            datapoints.append(datapoint_n)
        return datapoints

    def to_csv(self, path: str):
        df = pd.DataFrame(self.data)
        df.to_csv(path, index=False)

    def __iter__(self):
        for datapoint in self.data:
            yield (datapoint[field] for field in self.fields)

    def __len__(self):
        return len(self.data)

    def _sanity_check(self, other: KeyValueDataset):
        assert (
            self.data[0].keys() == other.data[0].keys()
        ), "You cannot do dataset operation on datasets with different keys"
        assert self.task_type == other.task_type, "You cannot do dataset operation on datasets for different tasks"
        assert len(self.fields) == len(
            other.fields
        ), "You cannot do dataset operation on datasets with different number fields"

    def _data2identifier(self, data: List[str]):
        id2datapoint = {}
        identifier2id = {}
        identifiers = []
        for idx, datapoint in enumerate(data):
            id2datapoint[idx] = datapoint
            # "|||" is a naive separator
            identifier = "|||".join([datapoint[field] for field in self.fields])
            identifiers.append(identifier)
            identifier2id[identifier] = idx
        identifiers = set(identifiers)  # remove duplicates
        return id2datapoint, identifier2id, identifiers

    def _identifier2data(self, id2datapoint, identifier2id, identifiers):
        result_data = []
        for identifier in identifiers:
            result_data.append(id2datapoint[identifier2id[identifier]])
        return result_data

    def __or__(self, other: KeyValueDataset) -> KeyValueDataset:
        self._sanity_check(other)
        id2datapoint, identifier2id, identifiers = self._data2identifier(self.data + other.data)
        data = self._identifier2data(id2datapoint, identifier2id, identifiers)
        return KeyValueDataset(data, self.task_type, self.fields, self.perturbation_rate)

    def __and__(self, other: KeyValueDataset) -> KeyValueDataset:
        self._sanity_check(other)
        id2datapoint, identifier2id, identifiers = self._data2identifier(self.data)
        _, _, identifiers2 = self._data2identifier(other.data)

        identifiers = identifiers.intersection(identifiers2)
        data = self._identifier2data(id2datapoint, identifier2id, identifiers)
        return KeyValueDataset(data, self.task_type, self.fields, self.perturbation_rate)

    def __sub__(self, other: KeyValueDataset) -> KeyValueDataset:
        self._sanity_check(other)
        id2datapoint, identifier2id, identifiers = self._data2identifier(self.data)
        _, _, identifiers2 = self._data2identifier(other.data)

        identifiers = identifiers.difference(identifiers2)
        data = self._identifier2data(id2datapoint, identifier2id, identifiers)
        return KeyValueDataset(data, self.task_type, self.fields, self.perturbation_rate)
