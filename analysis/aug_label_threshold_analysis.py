from tqdm import tqdm
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from training.generate_text_classification import label_with_semantic_preservation_and_lm_uniform_threshold
from transformations.no_transform import NoTransform


def experiment(aug_name: str):
    data_path = f"../dump/{aug_name}/wiki1m_for_simcse_train_100.csv"
    df = pd.read_csv(data_path)
    op = NoTransform()
    result_df = []
    pbar = tqdm(total=100)
    for ppl_t in np.linspace(0.3, 3, num=10):
        for sim_t in np.linspace(1.0, 10, num=10):
            df, ppl_neg, sim_neg = label_with_semantic_preservation_and_lm_uniform_threshold(
                df, op, ppl_t, sim_t, return_stat=True
            )
            pos_percent = sum(df.aug_labels) / len(df) * 100
            result_df.append(
                {
                    "positives": pos_percent,
                    "perplexity_threshold": round(ppl_t, 2),
                    "similarity_threshold": round(sim_t, 2),
                    "ppl_negatives_percent": ppl_neg,
                    "similarity_negative_percent": sim_neg,
                }
            )
            pbar.update(1)
    result_df = pd.DataFrame(result_df)
    sq_df = pd.pivot_table(
        result_df, values="positives", index=["perplexity_threshold"], columns=["similarity_threshold"], aggfunc=np.sum
    )
    ax = sns.heatmap(sq_df, annot=True)
    plt.title(f"Positive Augmentation Label (Percentage) for {aug_name}")
    plt.show()
    plt.savefig(f"../dump/{aug_name}/aug_positivity_heatmap.png")


def check_examples(aug_name: str, ppl_t: float, sim_t: float):
    data_path = f"../dump/{aug_name}/wiki1m_for_simcse_train_100.csv"
    df = pd.read_csv(data_path)
    op = NoTransform()
    df, ppl_neg, sim_neg = label_with_semantic_preservation_and_lm_uniform_threshold(
        df, op, ppl_t, sim_t, return_stat=True
    )
    df_pos = df[df.aug_labels]
    pass


if __name__ == "__main__":
    # experiment("RandomCrop")
    experiment("DiverseParaphrase")
    # experiment("Summarization")

    # check_examples("RandomCrop", 2.0, 5.5)
