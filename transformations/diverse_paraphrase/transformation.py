import random
from typing import List, Union

import numpy as np
import torch
from random import sample
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType

from transformations.diverse_paraphrase.submod.submodopt import SubmodularOpt
from transformations.diverse_paraphrase.submod.submodular_funcs import trigger_dips


class DiverseParaphrase(SentenceOperation):
    tasks = [TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_TO_TEXT_GENERATION]
    languages = ["en"]

    def __init__(self, augmenter="dips", num_outputs=1, seed=42):
        super().__init__()

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert augmenter in ["dips", "random", "diverse_beam", "beam"]
        if self.verbose:
            choices = ["dips", "random", "diverse_beam", "beam"]
            print(
                "The base paraphraser being used is Backtranslation - Generating {} candidates based on {}\n".format(
                    num_outputs, augmenter
                )
            )
            print("Primary options for augmenter : {}. \n".format(str(choices)))
            print(
                "Default: augmenter='dips', num_outputs=3. Change using DiverseParaphrase(augmenter=<option>, num_outputs=<num_outputs>)\n"
            )
            print("Starting to load English to German Translation Model.\n")

        name_en_de = "facebook/wmt19-en-de"
        self.tokenizer_en_de = FSMTTokenizer.from_pretrained(name_en_de)
        self.model_en_de = FSMTForConditionalGeneration.from_pretrained(name_en_de).to(self.device)

        if self.verbose:
            print("Completed loading English to German Translation Model.\n")
            print("Starting to load German to English Translation Model:")

        name_de_en = "facebook/wmt19-de-en"
        self.tokenizer_de_en = FSMTTokenizer.from_pretrained(name_de_en)
        self.model_de_en = FSMTForConditionalGeneration.from_pretrained(name_de_en).to(self.device)

        if self.verbose:
            print("Completed loading German to English Translation Model.\n")

        self.augmenter = augmenter
        if self.augmenter == "dips":
            if self.verbose:
                print("Loading word2vec gensim model. Please wait...")
            trigger_dips()
            if self.verbose:
                print("Completed loading word2vec gensim model.\n")
        self.num_outputs = num_outputs

    def en2de(self, input) -> str:
        input_ids = self.tokenizer_en_de.encode(input, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        outputs = self.model_en_de.generate(input_ids)
        decoded = self.tokenizer_en_de.decode(outputs[0], skip_special_tokens=True)
        if self.verbose:
            print(decoded)
        return decoded

    def en2de_batch(self, input: List[str]) -> List[str]:
        input_ids = self.tokenizer_en_de(input, return_tensors="pt", truncation=True, padding=True)
        input_ids = input_ids.to(self.device)
        outputs = self.model_en_de.generate(input_ids["input_ids"])
        decoded = self.tokenizer_en_de.batch_decode(outputs, skip_special_tokens=True)
        if self.verbose:
            print(decoded)
        return decoded

    def generate_diverse(self, en: Union[str, List[str]]) -> Union[List[str], List[List[str]]]:
        try:
            if isinstance(en, str):
                de = self.en2de(en)
                if self.augmenter == "diverse_beam":
                    en_new = self.generate_diverse_beam(de)
                else:
                    en_new = self.select_candidates(de, en)
            else:
                de_batch = self.en2de_batch(en)
                if self.augmenter == "diverse_beam":
                    # TODO yet to refactor
                    en_new = self.generate_diverse_beam(de_batch)
                else:
                    en_new = self.select_candidates_batch(de_batch, en)
        except Exception:
            if self.verbose:
                print("Returning Default due to Run Time Exception")
            if isinstance(en, str):
                en_new = [en for _ in range(self.num_outputs)]
            else:
                en_new = [[en[i] for _ in range(self.num_outputs)] for i in range(len(en))]
        return en_new

    def select_candidates(self, input: str, sentence: str):
        input_ids = self.tokenizer_de_en.encode(input, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        outputs = self.model_de_en.generate(
            input_ids,
            num_return_sequences=self.num_outputs * 5,
            num_beams=self.num_outputs * 5,
        )

        predicted_outputs = []
        decoded = [self.tokenizer_de_en.decode(output, skip_special_tokens=True) for output in outputs]
        if self.augmenter == "dips":
            try:
                subopt = SubmodularOpt(decoded, sentence)
                subopt.initialize_function(0.4, a1=0.5, a2=0.5, b1=1.0, b2=1.0)
                predicted_outputs = list(subopt.maximize_func(self.num_outputs))
            except Exception as e:
                if self.verbose:
                    print("Error in SubmodularOpt: {}".format(e))
                predicted_outputs = decoded[: self.num_outputs]
        elif self.augmenter == "random":
            predicted_outputs = sample(decoded, self.num_outputs)
        else:  # Fallback to top n points in beam search
            predicted_outputs = decoded[: self.num_outputs]

        if self.verbose:
            print(predicted_outputs)

        return predicted_outputs

    def select_candidates_batch(self, input: List[str], sentences: List[str]):
        input_ids = self.tokenizer_de_en(input, padding=True, truncation=True, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        outputs = self.model_de_en.generate(
            input_ids.input_ids,
            num_return_sequences=self.num_outputs * 5,
            num_beams=self.num_outputs * 5,
        )
        predicted_outputs_batch = []
        decoded_batch = self.tokenizer_de_en.batch_decode(outputs, skip_special_tokens=True)
        for idx, sentence in enumerate(sentences):
            decoded = decoded_batch[idx * self.num_outputs * 5 : (idx + 1) * self.num_outputs * 5]
            if self.augmenter == "dips":
                try:
                    subopt = SubmodularOpt(decoded, sentence)
                    subopt.initialize_function(0.4, a1=0.5, a2=0.5, b1=1.0, b2=1.0)
                    predicted_outputs = list(subopt.maximize_func(self.num_outputs))
                except Exception as e:
                    if self.verbose:
                        print("Error in SubmodularOpt: {}".format(e))
                    predicted_outputs = decoded[: self.num_outputs]
            elif self.augmenter == "random":
                predicted_outputs = sample(decoded, self.num_outputs)
            else:  # Fallback to top n points in beam search
                predicted_outputs = decoded[: self.num_outputs]
            predicted_outputs_batch.append(predicted_outputs)

        if self.verbose:
            print(predicted_outputs_batch)
        assert len(predicted_outputs_batch) == len(sentences)
        return predicted_outputs_batch

    def generate_diverse_beam(self, sentence: str):
        input_ids = self.tokenizer_de_en.encode(sentence, return_tensors="pt")
        input_ids = input_ids.to(self.device)

        try:
            outputs = self.model_de_en.generate(
                input_ids,
                num_return_sequences=self.num_outputs,
                num_beam_groups=2,
                num_beams=self.num_outputs,
            )
        except:
            outputs = self.model_de_en.generate(
                input_ids,
                num_return_sequences=self.num_outputs,
                num_beam_groups=1,
                num_beams=self.num_outputs,
            )

        predicted_outputs = [self.tokenizer_de_en.decode(output, skip_special_tokens=True) for output in outputs]

        if self.verbose:
            print(predicted_outputs)

        return predicted_outputs

    def generate(self, sentence: str) -> List[str]:
        candidates = self.generate_diverse(sentence)
        return candidates

    def generate_batch(self, sentences: List[str]) -> List[List[str]]:
        candidates_batch = self.generate_diverse(sentences)
        return candidates_batch
