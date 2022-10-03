import random
from typing import List

import nlpaug.augmenter.word as naw
import torch.cuda

from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType

"""
Base Class for implementing the different input transformations a generation should be robust against.
"""


class RandomContextualWordAugmentation(SentenceOperation):
    tasks = [TaskType.TEXT_CLASSIFICATION, TaskType.TEXT_TO_TEXT_GENERATION]
    languages = ["en"]
    keywords = []

    def __init__(self, prob=0.3, seed=0):
        super().__init__(seed)
        self.prob = prob
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.augs = [
            naw.ContextualWordEmbsAug(
                model_path="distilbert-base-uncased", action="insert", aug_p=prob, device=self.device
            ),
            naw.ContextualWordEmbsAug(
                model_path="distilbert-base-uncased", action="substitute", aug_p=prob, device=self.device
            ),
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
