from datasets import load_dataset
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
import torch
import os


class TestDataset:
    def __init__(self, base_dir="/project"):
        self.base_dir = base_dir

    def read_dataset(self, path=None, config_name=None, split_type=None):
        os.makedirs(pjoin(self.base_dir, "data"), exist_ok=True)
        if path != None:
            if config_name == None:
                self.dataset = load_dataset(path, split=split_type, cache_dir=pjoin(self.base_dir, "data"))
            else:
                self.dataset = load_dataset(path, config_name, split=split_type, cache_dir=pjoin(self.base_dir, "data"))

    def calculate_metrics():
        pass


class TripletDataset(Dataset):
    def __init__(self, refs, pos_examples, neg_examples):
        """
        Args:
            X: The unicode version of names
            y: Race labels for the names
        """
        self.refs = refs
        self.pos_examples = pos_examples
        self.neg_examples = neg_examples

    def __len__(self):
        return len(self.refs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        name = torch.from_numpy(self.X[idx])
        label = torch.from_numpy(np.array(self.y[idx]))
        label = label.type(torch.LongTensor)
        sample = {"name": name, "label": label}
        return sample


"""
MNLI_dataset
    sts_test 

parallel_dataset
    sts_test

xyz_dataset
    sts_test 


STS_Experiments
    for any dataset that has sts_test
        run sts_test with my model and tokenizer loaded. So the dataset methods should 
        take tokenizers externally. 
        Save outputs for each test
    save overall test Parameters and basic results

models are more homogenous
datasets have more change

option II

STS_Experiments
    load model
    sts_test
        run the same test for each dataset
        here datasets don't need to have any unique class I can just write the experiments
        and sts mnli etc experiments class and whenever we want to add any additional
        custom datasets etc we just have to write them in as 


task class: class that runs the experiments once a dataset of this class is transformed into a certain 
structure. Each instance of this class should have a preprocess function that takes care of the 
these transformations.

The task class should also include generic actions like loading the dataset and calculating 
some metrics when and if necesary

Experiment class: should take in a model a list of dataset classes and run the type of experiments
that the the model requires on all dataset classes iteratively. Log the results as both macro logs
and as experiment results. 
"""
