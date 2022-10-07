# if no positive augmentation is provided and is needed in the loss, default
# positive augmentation is same sentence with different dropout (SimCSE)
# The following script trains a model that uses original sentence as positives
# and Random Deletion as negatives.

python train.py\
        --model_name_or_path "bert-base-uncased"\
        --pooler_type "cls"\
        --contrastive_loss_type "ranking_pos_neg"\
        --temp 0.05\
        --hard_negative_weight 0\
        --dataset_name "/projectnb/llamagrp/peter/ContrastiveAugmentation/data/wiki1m_for_simcse.txt"\
        --positive_augmentations ""\
        --negative_augmentations "RandomDeletion"\
        --regenerate_augmentation_per_epoch false\
        --max_seq_length 32\
        --uniform_augmentation_sampling false\
        --remove_no_augmentations true\
        --output_dir dump\
        --overwrite_output_dir\
        --overwrite_cache\
        --do_train true\
        --do_eval true\
        --evaluation_strategy steps\
        --eval_steps 1000\
        --eval_original false\
        --eval_transfer true\
        --eval_robust true\
        --learning_rate $LR\
        --per_device_train_batch_size 128\
        --per_device_eval_batch_size 4\
        --num_train_epochs 1\
        --remove_unused_columns false\
        --report_to wandb\
        --wandb_project "ContrastiveAugmentation"\
        --logging_steps 500\
        --fp16\
        --save_strategy epoch\
        --hyper_path_modifier debug
