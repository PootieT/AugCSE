import os
import random
from typing import List

import numpy as np
import nlpaug.augmenter.word as naw

from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType

"""
Base Class for implementing the different input transformations a generation should be robust against.
"""


class RandomWordEmbAugmentation(SentenceOperation):
    tasks = [TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_TO_TEXT_GENERATION]
    languages = ["en"]
    keywords = []

    def __init__(self, prob=0.3, glove_dim: int = 50, seed=0):
        """
        glove_dim: one of [50, 100, 200, 300]
        """
        super().__init__(seed)
        self.prob = prob
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        # model_path = f"{curr_dir}/cc.en.300.bin"
        model_path = f"{curr_dir}/glove.6B/glove.6B.{glove_dim}d.txt"
        if not os.path.isfile(model_path):
            import urllib.request
            import zipfile

            # hasn't gotten fasttext to work
            # urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz")
            urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.6B.zip")

            with zipfile.ZipFile(f"{curr_dir}/glove.6B.zip", "r") as zip_ref:
                zip_ref.extractall(f"{curr_dir}/glove.6B")
        self.augs = [
            naw.WordEmbsAug(model_type="glove", model_path=model_path, action="insert", aug_p=prob),
            naw.WordEmbsAug(model_type="glove", model_path=model_path, action="substitute", aug_p=prob),
        ]

    def generate(self, sentence: str) -> List[str]:
        augmented_text = random.choice(self.augs).augment(sentence)
        return [augmented_text]


"""
# Sample code to demonstrate usage. Can also assist in adding test cases.
# You don't need to keep this code in your transformation.
if __name__ == '__main__':
    import json
    from TestRunner import convert_to_snake_case

    tf = RandomDeletion()
    sentence = "Andrew finally returned the French book to Chris that I bought last week"
    test_cases = []
    for sentence in ["Andrew finally returned the French book to Chris that I bought last week",
                     "Sentences with gapping, such as Paul likes coffee and Mary tea, lack an overt predicate to indicate the relation between two or more arguments.",
                     "Alice in Wonderland is a 2010 American live-action/animated dark fantasy adventure film",
                     "Ujjal Dev Dosanjh served as 33rd Premier of British Columbia from 2000 to 2001",
                     "Neuroplasticity is a continuous processing allowing short-term, medium-term, and long-term remodeling of the neuronosynaptic organization."]:
        test_cases.append({
            "class": tf.name(),
            "inputs": {"sentence": sentence}, "outputs": [{"sentence": o} for o in tf.generate(sentence)]}
        )
    json_file = {"type": convert_to_snake_case(tf.name()), "test_cases": test_cases}
    with open("test.json", "w") as f:
        ijson.dump(json_file, f, indent=2)
"""
