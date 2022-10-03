from testing_part.src.test_datasets.test_dataset import TestDataset


class NLIDataset(TestDataset):
    def __init__(self, base_dir="/project"):
        self.base_dir = base_dir

    def preprocess_pair(self):
        self.read_dataset(path="anli", config_name="plain_text", split_type="train_r1")
        premises = {}
        for i, example in enumerate(self.dataset):
            temp_list = premises.get(example["premise"], [])
            temp_list.append((i, example["label"]))
            premises[example["premise"]] = temp_list
        refs = []
        pos_samples = []
        neg_samples = []
        for ref, hyps_list in premises.items():
            for tup in hyps_list:
                for tup2 in hyps_list:
                    if tup[1] == 0:
                        if tup2[1] != 0:
                            refs.append(ref)
                            pos_samples.append(self.dataset[tup[0]]["hypothesis"])
                            neg_samples.append(self.dataset[tup2[0]]["hypothesis"])
        return refs, pos_samples, neg_samples
