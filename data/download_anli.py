import numpy as np
import pandas as pd
import datasets
from datasets import load_dataset


def sample_with_default(s: pd.Series) -> str:
    return np.nan if len(s) == 0 else s.sample()


def download_and_process_anli(rounds: str):
    df_all = pd.DataFrame()
    for r in ["1", "2", "3"] if rounds == "all" else [rounds]:
        dataset = load_dataset("anli", split=f"train_r{r}")
        df = pd.DataFrame(dataset)
        df = df.drop(columns=["uid", "reason"])
        df_pivot = pd.pivot_table(
            df, values="hypothesis", index="premise", columns="label", aggfunc=sample_with_default
        )
        df_pivot = df_pivot.drop(columns=[1])
        df_pivot = df_pivot[(~df_pivot[0].isnull()) & (~df_pivot[2].isnull())]
        df_pivot.reset_index(inplace=True)
        df_pivot = df_pivot.rename(columns={"premise": "sent0", 0: "sent1", 2: "hard_neg"})
        df_all = df_all.append(df_pivot)
    df_all.to_csv(f"anli_{rounds}_for_simcse.csv", index=False)


def download_and_process_anli_pivot_all(rounds: str):
    df_all = pd.DataFrame()
    for r in ["1", "2", "3"] if rounds == "all" else [rounds]:
        dataset = load_dataset("anli", split=f"train_r{r}")
        df = pd.DataFrame(dataset)
        df = df.drop(columns=["uid", "reason"])
        df_all = df_all.append(df)
    df_pivot = pd.pivot_table(
        df_all, values="hypothesis", index="premise", columns="label", aggfunc=sample_with_default
    )
    df_pivot = df_pivot.drop(columns=[1])
    df_pivot = df_pivot[(~df_pivot[0].isnull()) & (~df_pivot[2].isnull())]
    df_pivot.reset_index(inplace=True)
    df_pivot = df_pivot.rename(columns={"premise": "sent0", 0: "sent1", 2: "hard_neg"})
    df_pivot.to_csv(f"anli_{rounds}_for_simcse.csv", index=False)


if __name__ == "__main__":
    np.random.seed(42)
    # download_and_process_anli("all")
    # remove repeated premises
    # download_and_process_anli_pivot_all("all")
