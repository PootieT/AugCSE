# Dataset (Unperturbed)

In here we included original datasets from [SimCSE](https://github.com/princeton-nlp/SimCSE/tree/main/data)
as well as the dataset we procured from [ANLI](https://huggingface.co/datasets/anli)
in the format of our training data.

To download the wiki1m or NLI, simply run:
```commandline
./download_nli.sh
./download_wiki.sh
```

To download ANLI data, simply run:
```commandline
python download_anli.py
```

For augmented dataset, please download them through this [link](https://drive.google.com/file/d/1TZjscSzUSLnUDDO4caxKR8T6K8RcxxUj/view?usp=sharing)
Once downloaded, unzip the `dump` folder and place it under the project directory `AugCSE/dump`.