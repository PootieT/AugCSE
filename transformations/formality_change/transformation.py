print("===============> loading formality change transformations!")

import random
from typing import List

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
)
import torch
from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType

print("===============> finished importing libraries from formality change transformations!")
"""
code and model from: https://github.com/PrithivirajDamodaran/Styleformer
"""


class Adequacy:
    def __init__(self, model_tag="prithivida/parrot_adequacy_on_BART"):
        print("=============> initializing adequacy class")
        # lazy loading models so during training, when we don't need these models, network
        # errors from huggingface do not prevent us from initializing jobs
        self.model_tag = model_tag
        self._nli_model = None
        self._tokenizer = None
        print("=============> finished init adequacy class")

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_tag)
        return self._tokenizer

    @property
    def nli_model(self):
        if self._nli_model is None:
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(self.model_tag)
        return self._nli_model

    def filter(self, input_phrase, para_phrases, adequacy_threshold, device="cpu"):
        top_adequacy_phrases = []
        for para_phrase in para_phrases:
            x = self.tokenizer.encode(
                input_phrase,
                para_phrase,
                return_tensors="pt",
                truncation_strategy="only_first",
            )
            self.nli_model = self.nli_model.to(device)
            logits = self.nli_model(x.to(device))[0]
            # we throw away "neutral" (dim 1) and take the probability of "entailment" (2) as the adequacy score
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            adequacy_score = prob_label_is_true[0].item()
            if adequacy_score >= adequacy_threshold:
                top_adequacy_phrases.append(para_phrase)
        return top_adequacy_phrases

    def score(self, input_phrase, para_phrases, adequacy_threshold, device="cpu"):
        adequacy_scores = {}
        for para_phrase in para_phrases:
            x = self.tokenizer.encode(
                input_phrase,
                para_phrase,
                return_tensors="pt",
                truncation_strategy="only_first",
            )
            self.nli_model = self.nli_model.to(device)
            logits = self.nli_model(x.to(device))[0]
            # we throw away "neutral" (dim 1) and take the probability of "entailment" (2) as the adequacy score
            entail_contradiction_logits = logits[:, [0, 2]]
            probs = entail_contradiction_logits.softmax(dim=1)
            prob_label_is_true = probs[:, 1]
            adequacy_score = prob_label_is_true[0].item()
            if adequacy_score >= adequacy_threshold:
                adequacy_scores[para_phrase] = adequacy_score
        return adequacy_scores

    def score_batch(self, input_phrase, para_phrases, adequacy_threshold, device="cpu"):
        input_pairs = [(input_phrase, para_phrase) for para_phrase in para_phrases]

        x = self.tokenizer(
            input_pairs,
            return_tensors="pt",
            padding=True,
            truncation_strategy="only_first",
        )
        self.nli_model = self.nli_model.to(device)
        logits = self.nli_model(x.to(device).input_ids)[0]
        # we throw away "neutral" (dim 1) and take the probability of "entailment" (2) as the adequacy score
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        prob_label_is_true = probs[:, 1]
        adequacy_scores = {
            para_phrases[i]: score for i, score in enumerate(prob_label_is_true.cpu().detach().numpy().tolist())
        }
        return adequacy_scores


adequacy = Adequacy()


class Formal2Casual(SentenceOperation):
    tasks = [TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_TO_TEXT_GENERATION]
    languages = ["en"]
    heavy = True

    def __init__(self, num_beams=5, max_length=32, quality_filter=0.95, device=None):
        super().__init__()
        if self.verbose:
            print("Starting to load Casual to Formal Model...\n")
        m_name = "prithivida/formal_to_informal_styletransfer"
        self.tokenizer = AutoTokenizer.from_pretrained(m_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(m_name)
        if self.verbose:
            print("Completed loading Casual to Formal Model.\n")
        self.adequacy = adequacy
        self.max_output = num_beams
        self.num_beams = num_beams
        self.max_length = max_length
        self.quality_filter = quality_filter
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, sentence: str):
        ctf_prefix = "transfer Formal to Casual: "
        src_sentence = sentence
        sentence = ctf_prefix + sentence
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        model = self.model.to(self.device)
        input_ids = input_ids.to(self.device)

        preds = model.generate(
            input_ids,
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True,
            num_return_sequences=self.max_output,
        )
        model = self.model.to("cpu")  # save GPU memory for scoring
        input_ids = input_ids.to("cpu")

        gen_sentences = set()
        for pred in preds:
            gen_sentences.add(self.tokenizer.decode(pred, skip_special_tokens=True).strip())

        try:
            adequacy_scored_phrases = self.adequacy.score(
                src_sentence, list(gen_sentences), self.quality_filter, self.device
            )
        except:
            # in case CUDA out of memory, just randomly assign adquacy score
            adequacy_scored_phrases = {s: random.uniform(0, 1) for s in gen_sentences}
        ranked_sentences = sorted(adequacy_scored_phrases.items(), key=lambda x: x[1], reverse=True)
        if len(ranked_sentences) > 0:
            return [ranked_sentences[0][0]]
        else:
            print("No transfer found!")
            return [sentence]


class Casual2Formal(SentenceOperation):
    tasks = [TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_TO_TEXT_GENERATION]
    languages = ["en"]
    heavy = True

    def __init__(self, num_beams=5, max_length=32, quality_filter=0.95, device=None):
        super().__init__()
        if self.verbose:
            print("Starting to load Casual to Formal Model...\n")
        m_name = "prithivida/informal_to_formal_styletransfer"
        self.tokenizer = AutoTokenizer.from_pretrained(m_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(m_name)
        if self.verbose:
            print("Completed loading Casual to Formal Model.\n")
        self.adequacy = adequacy
        self.max_output = num_beams
        self.num_beams = num_beams
        self.max_length = max_length
        self.quality_filter = quality_filter
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate(self, sentence: str):
        ctf_prefix = "transfer Casual to Formal: "
        src_sentence = sentence
        sentence = ctf_prefix + sentence
        input_ids = self.tokenizer.encode(sentence, return_tensors="pt")
        model = self.model.to(self.device)
        input_ids = input_ids.to(self.device)

        preds = model.generate(
            input_ids,
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True,
            num_return_sequences=self.max_output,
        )
        model = self.model.to("cpu")  # save GPU memory for scoring
        input_ids = input_ids.to("cpu")

        gen_sentences = set()
        for pred in preds:
            gen_sentences.add(self.tokenizer.decode(pred, skip_special_tokens=True).strip())

        try:
            adequacy_scored_phrases = self.adequacy.score(
                src_sentence, list(gen_sentences), self.quality_filter, self.device
            )
        except:
            # in case CUDA out of memory, just randomly assign adquacy score
            adequacy_scored_phrases = {s: random.uniform(0, 1) for s in gen_sentences}

        ranked_sentences = sorted(adequacy_scored_phrases.items(), key=lambda x: x[1], reverse=True)
        if len(ranked_sentences) > 0:
            return [ranked_sentences[0][0]]
        else:
            return [sentence]

    def generate_batch(self, sentences: List[str]) -> List[List[str]]:
        ctf_prefix = "transfer Casual to Formal: "
        src_sentences = sentences
        sentences = [ctf_prefix + s for s in sentences]
        input_ids = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        model = self.model.to(self.device)
        input_ids = input_ids.to(self.device)

        preds = model.generate(
            input_ids.input_ids,
            num_beams=self.num_beams,
            max_length=self.max_length,
            early_stopping=True,
            num_return_sequences=self.max_output,
        )
        model = self.model.to("cpu")  # save GPU memory for scoring
        input_ids = input_ids.to("cpu")
        output_sentences = []
        for i, src_sentence in enumerate(src_sentences):
            gen_sentences = set()
            for pred in preds[i * self.max_output : (i + 1) * self.max_output]:
                gen_sentences.add(self.tokenizer.decode(pred, skip_special_tokens=True).strip())

            try:
                adequacy_scored_phrases = self.adequacy.score_batch(
                    src_sentence, list(gen_sentences), self.quality_filter, self.device
                )
            except:
                # in case CUDA out of memory, just randomly assign adquacy score
                adequacy_scored_phrases = {s: random.uniform(0, 1) for s in gen_sentences}

            ranked_sentences = sorted(adequacy_scored_phrases.items(), key=lambda x: x[1], reverse=True)
            if len(ranked_sentences) > 0:
                output_sentence = [ranked_sentences[0][0]]
            else:
                output_sentence = [src_sentence]
            output_sentences.append(output_sentence)
        return output_sentences
