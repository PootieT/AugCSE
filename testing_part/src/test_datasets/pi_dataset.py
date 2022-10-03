from testing_part.src.test_datasets.test_dataset import TestDataset


class PIDataset(TestDataset):
    def __init__(self, base_dir="/project"):
        self.base_dir = base_dir

    def preprocess_pair(self):
        pass
