# AugCSE: Contrastive Sentence Embedding with Diverse Augmentation

This is the repo for paper `AugCSE: Contrastive Sentence Embedding with Diverse Augmentation` - AACL 2022.
We used a diverse set of natural language augmentation and a discriminator loss to improve general sentence embedding 
representation (in English). We show that simple rule-based augmentations can be just as effective at improving sentence
representations than language-model based augmentations, which can be expensive and biased.

![Architecture](architecture.png)

In this repo, you will find 
- 100+ natural language augmentations (from NL-Augmenter)
- Training [script](./train.py) for AugCSE with different options(see [training arguments](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L64)):
  - loss variations (triplet, ranking (SimCSE), contrastive, etc) [link](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L126)
  - projection layer variations (linear, mixture of experts) [link](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L104)
  - discriminator variations (binary, predict label, predict order) [link](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L171)
  - positive, negative, or neutral augmentations [link](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L265)
  - augmentation automatic labeling and sampling variations [link](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L296)
  - supervised vs. unsupervised objective [link](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L127)
  - single sentence vs. pair sentence classification [link](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L216)
- Evaluation [script](./evaluate_sentence_embedding.py) for STS and SentEval (adapted from SimCSE)
- Various experimentation and analysis [scripts](./analysis)

If you have any questions, please contact `zilutang@bu.edu`, or submit issues / PRs.


## Environment

```commandline
conda create --name augmentation python=3.8
conda activate augmentation
pip install -r requirements.txt
```

## Data

Original as well as augmentation data for wiki1m can be downloaded. Read more [here](./data/README_2.md)
For augmented data, once downloaded, the folder of the zip file (unzipped) should be placed at the root of the project
for training script to work property:
```commandline
AugCSE/dump/{AUGMENTATION}/wiki1m_for_simcse_train_100.csv
```


## Train
mention this is our model loss, if you want to use new augs, what would you do
To reproduce our results (for STS) with AugCSE framework, run the following training [script](./experiments/run_train_exp.py):
```commandline
./experiments/run_train_exp.py
```

The main arguments are explained as follows (for more detailed explanation, see actual [arguments](https://github.com/PootieT/AugCSE/blob/39c7ee902c635b189528cf3f860e42e66a4a8078/train.py#L64)):
```commandline
--model_name_or_paths "bert-base-uncased"   # this allows you to switch to any backbone 
                                            # models supported in huggingface.
```


To train a model with positive augmentation as original sentence (SimCSE objective) and RandomDeletion
as negative augmentation, try:
```commandline
./experiments/run_train_pos_neg_simcse.py
```

### mention how user can change augmentation names, create custom augmentations, as well as using multiples
also easy to swap model name, add the argument difference examples, what they want, and how they can do it

## Evaluate
For example evaluation on STS or SentEval, first download SentEval dataset
```commandline
cd SentEval/data/downstream && ./download_dataset.sh
```

Then, run evaluation script: 
```commandline
./run_eval.sh
```

## Hyperparameters:

For best hyperparameters, please refer to paper Appendix A.7 for detail.



## Future Studies:

This paper only shows the beginning of usefulness of rule based augmentations (especially in low-resource settings). A 
robust domain-agnostic sentence embedding can be useful for many downstream task, from natural language understanding to
text generation. Specific rule based augmentations can also be extremely valuable in instilling invanriances within the 
model, which becomes especially valuable in AI fairness (discriminative loss vs. instance contrastive loss also relates
nicely to improving group and individual fairness). If you want to collaborate on any of the above areas, feel free to 
shoot us an email! We are excited to hear how our repo can be made useful for your experimentation!


To cite our paper, please use:
```
@article{Tang2022AugCSE,
  title={AugCSE: Contrastive Sentence Embedding with Diverse Augmentation},
  author={Tang, Zilu and Kocyigit, Muhammed Yusuf and Wijaya, Derry},
  journal={link coming},
  year={2022}
}

```

The repo is adapted from [NL-Augmenter](https://github.com/GEM-benchmark/NL-Augmenter) and [SimCSE](https://github.com/princeton-nlp/SimCSE)
