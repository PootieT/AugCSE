import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_metrics(csv_path: str):
    df = pd.read_csv(csv_path)
    drop_cols = [c for c in df.columns if "MAX" in c or "MIN" in c or "_step" in c]
    df = df.drop(columns=drop_cols)
    rename_dict = {}
    for col in df.columns:
        start = col.find("GRM=")
        end = col.find("LR=")
        new_name = col[start:end].replace("GRM", "alpha")
        if "-" in new_name:
            new_name = new_name.replace("-", "")
        else:
            new_name = new_name[:6] + "-" + new_name[6:]
        rename_dict[col] = new_name
    rename_dict["train/epoch"] = "Train Epoch"
    df = df.rename(columns=rename_dict)
    df = df[["Train Epoch", "alpha=-100.0", "alpha=-10.0", "alpha=-1.0", "alpha=1.0", "alpha=10.0", "alpha=100.0"]]
    df = pd.melt(df, ["Train Epoch"])
    df = df.rename(columns={"value": "Discriminator Accuracy"})
    sns.lineplot(data=df, x="Train Epoch", y="Discriminator Accuracy", hue="variable")
    plt.savefig("../dump/aggregate/alpha_discriminator_acc.png", bbox_inches="tight")
    pass


if __name__ == "__main__":
    plot_metrics("../dump/aggregate/alpha_discriminator_acc.csv")
