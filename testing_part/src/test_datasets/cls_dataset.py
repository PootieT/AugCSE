from posixpath import split
from testing_part.src.test_datasets.test_dataset import TestDataset


class CLSDataset(TestDataset):
    def __init__(
        self,
        base_dir="/project",
        path="glue",
        config_name="cola",
        split_type="validation",
    ):
        self.base_dir = base_dir
        self.read_dataset(path=path, config_name=config_name, split_type=split_type)

    def preprocess(self):
        pass
