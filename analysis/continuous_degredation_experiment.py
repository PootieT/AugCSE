import numpy as np
import pandas as pd
import torch.cuda
from sentence_transformers import SentenceTransformer
from transformers import pipeline

from evaluation.evaluate_similarity import _get_model_pred, similarity
from transformations.random_word_augmentation import *


def perturb(sentences: List[str], operation: str, **aug_kwargs) -> List[str]:
    augmentation = {"delete": RandomDeletion, "crop": RandomCrop, "swap": RandomSwap, "all": RandomWordAugmentation}[
        operation
    ](**aug_kwargs)

    result = []
    for sentence in sentences:
        result.append(augmentation.generate(sentence)[0])

    return result


def load_and_sample_sentences(path: str, sample_size: int = 1000) -> List[str]:
    df = pd.read_table(path, names=["text"])
    sentences = np.random.choice(df.text, size=sample_size, replace=False)
    np.random.shuffle(sentences)
    return sentences.tolist()


def calculate_similarities(sentences: List[str], pt_sentences: List[str], batch_size):
    # model_name = "nreimers/MiniLM-L6-H384-uncased"
    # model_name = "bert-base-uncased"
    # model_name = "sentence-transformers/bert-base-nli-mean-tokens"
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model = SentenceTransformer(model_name)
    similarities, similarity_samples = _get_model_pred(
        model,
        sentences,
        pt_sentences,
        batch_size=batch_size,
    )
    return similarities, similarity_samples


def get_similarities_from_embeddings(orig_emb_all, aug_emb_all):
    # d = aug_emb_all.device
    d = "cuda"
    similarities = []

    for i in range(0, len(orig_emb_all), 1000):
        orig_emb = torch.tensor(orig_emb_all[i : i + 1000], device=d)
        aug_emb = torch.tensor(aug_emb_all[i : i + 1000], device=d)

        orig_sim = similarity(orig_emb.unsqueeze(1), orig_emb.unsqueeze(0))
        aug_sim = similarity(aug_emb.unsqueeze(1), aug_emb.unsqueeze(0))
        cross_sim = similarity(orig_emb.unsqueeze(1), aug_emb.unsqueeze(0))

        actual_bs = len(orig_sim)
        # average similarity between a sample and a random sample
        orig_rand_sim = torch.sum(orig_sim - orig_sim * torch.eye(actual_bs, device=d)) / (actual_bs ** 2 - actual_bs)
        # average similarity between a augmented sample and a random augmented sample
        aug_rand_sim = torch.sum(aug_sim - aug_sim * torch.eye(actual_bs, device=d)) / (actual_bs ** 2 - actual_bs)

        # average similarity between the original and the augmented sample
        aug_self_sim = torch.sum(torch.diagonal(cross_sim, 0)) / actual_bs

        # average similarity between a sample and a random augmented sample
        aug_orig_rand_sim = torch.sum(cross_sim - cross_sim * torch.eye(actual_bs, device=d)) / (
            actual_bs ** 2 - actual_bs
        )
        similarities.append([orig_rand_sim, aug_rand_sim, aug_self_sim, aug_orig_rand_sim])
    similarities = torch.hstack([torch.vstack([i[0], i[1], i[2], i[3]]) for i in similarities]).mean(1)

    return {
        "orig_rand_sim": similarities[0].item(),
        "aug_rand_sim": similarities[1].item(),
        "aug_self_sim": similarities[2].item(),
        "aug_orig_rand_sim": similarities[3].item(),
    }


if __name__ == "__main__":
    operation = "delete"
    probs = np.linspace(0.1, 0.9, num=9)
    sims = []
    sents = load_and_sample_sentences("../data/wiki1m_for_simcse.txt")
    for prob in probs:
        pt_sents = perturb(sents, operation, prob=prob, aug_max=30)
        sim, _ = calculate_similarities(sents, pt_sents, batch_size=128)
        sim["prob"] = prob
        sims.append(sim)

    sim_df = pd.DataFrame(sims)
    pass
