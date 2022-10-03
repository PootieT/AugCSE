from interfaces.SentenceOperation import SentenceOperation
from tasks.TaskTypes import TaskType

"""
Base Class for implementing the different input transformations a generation should be robust against.
"""


class NoTransform(SentenceOperation):
    """
    placeholder transformation class for datasets that don't need transformation

    """

    tasks = [
        TaskType.TEXT_CLASSIFICATION,
    ]
    languages = ["en"]
    keywords = []

    def __init__(self, seed=0, max_outputs=1):
        super().__init__(seed, max_outputs=max_outputs)

    def generate(self, sentence: str):
        return [sentence]
