from tasks.TaskTypes import TaskType
from train import ModelArguments, DataTrainingArguments, OurTrainingArguments, main


# The following script is an example to invoke training script through python.
# (useful in debugging with IDEs like Pycharm)
# Below code should be able to reproduce our results for STS-B performance. For
# Detailed hyperparameter settings, please refer to paper appendix A.7

model_args = ModelArguments(
    model_name_or_path="bert-base-uncased",  # "roberta-base", or any other model supported by huggingface
    pooler_type="cls",
    projection_layers=100,  # a short-cut for predefined 2 layer mlp projection layer
    supervised_augmentation_loss=False,
    contrastive_loss_type="ranking",         # follows SBERT loss naming convention
    discriminate=True,
    discriminator_weight=0.005,
    gradient_reverse_multiplier=1.0,
    diff_cse_mode=True
)
data_args = DataTrainingArguments(
    dataset_name="../data/wiki1m_for_simcse.txt",
    overwrite_cache=True,
    positive_augmentations="ContractionExpansions,DiverseParaphrase,Casual2Formal,SentenceAuxiliaryNegationRemoval,TenseTransformation,DiscourseMarkerSubstitution,SentenceSubjectObjectSwitch,RandomSwap,CityNamesTransformation,SentenceAdjectivesAntonymsSwitch,YodaPerturbation,SentenceReordering",
    augmentation_label_method="lm_uniform",
    regenerate_augmentation_per_epoch=False,
    resample_augmentation_per_epoch=False,
    sample_default_augmentations=True,
    task_type=TaskType.TEXT_CLASSIFICATION,
    force_regenerate=False,
    max_seq_length=32,
    uniform_augmentation_sampling=True,
    remove_no_augmentations=False,
)
training_args = OurTrainingArguments(
    output_dir="dump",  # this will be modified in script
    overwrite_output_dir=True,
    hyper_path_modifier="baseline",
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=500,
    eval_original=False,  # eval on original evaluation set or sts and/or senteval
    eval_transfer=True,
    eval_robust=False,
    learning_rate=2e-5,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=1,
    num_train_epochs=4,
    remove_unused_columns=False,
    report_to="wandb",
    wandb_project="ContrastiveAugmentation",
    logging_steps=500,
    save_strategy="steps",  # "steps",
    save_steps=500,
    load_best_model_at_end=True,
    metric_for_best_model="stsb_spearman,avg_transfer",  # "avg_transfer"
    seed=42,
)
main(model_args, data_args, training_args)
