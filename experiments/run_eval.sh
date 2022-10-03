CHECKPOINT="path/to/checkpoint/folder"

python evaluate_sentence_embedding.py\
  --model_name_or_path bert-base-uncased\
  --model_checkpoint $CHECKPOINT\
  --pooler avg\
  --mode test\
  --task_set sts