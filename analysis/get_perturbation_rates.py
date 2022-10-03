import os
from os.path import dirname, abspath
from typing import Dict
import pprint

from tqdm import tqdm
import pandas as pd


def collect_perturbation_rates(path: str, data_file_name: str) -> Dict[str, float]:
    result = {}
    for aug in tqdm(os.listdir(path)):
        file_path = f"{path}/{aug}/{data_file_name}"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if "sent1_aug" in df.columns:
                pt_rate = sum((df.sent1_aug.isna()) | (df.sent1 == df.sent1_aug) | (df.sent1_aug == "")) / len(df)
                result[aug] = 1 - pt_rate
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)
    return result


if __name__ == "__main__":
    dump_path = dirname(dirname(abspath(__file__))) + "/dump"
    data_file_str = "wiki1m_for_simcse_train_100.csv"
    collect_perturbation_rates(dump_path, data_file_str)
