# Adapted from https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py


from typing import Optional

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from training.discriminator import AugmentationDiscriminator
from training.mlp import ProjectionMLP, MLPLayer
from training.moe import MoE


class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """

    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "avg", "avg_top2", "avg_first_last"], (
            "unrecognized pooling type %s" % self.pooler_type
        )

    def forward(self, attention_mask, outputs):
        last_hidden = outputs.hidden_states[-1]
        hidden_states = outputs.hidden_states

        if self.pooler_type == "cls":
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return (last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1)
        elif self.pooler_type == "avg_first_last":
            first_hidden = hidden_states[0]
            last_hidden = hidden_states[-1]
            pooled_result = ((first_hidden + last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        elif self.pooler_type == "avg_top2":
            second_last_hidden = hidden_states[-2]
            last_hidden = hidden_states[-1]
            pooled_result = ((last_hidden + second_last_hidden) / 2.0 * attention_mask.unsqueeze(-1)).sum(
                1
            ) / attention_mask.sum(-1).unsqueeze(-1)
            return pooled_result
        else:
            raise NotImplementedError


class ContrastiveModel(nn.Module):
    def __init__(self, transformer_model):
        super().__init__()
        self.transformer_model = transformer_model
        self.config = transformer_model.config
        self.pooler = Pooler(self.config.pooler_type)
        if self.config.moe and self.config.projection_heads > 1 and self.config.projection_layers == 100:
            self.projection_layer = MoE(
                input_size=self.config.hidden_size,
                num_experts=self.config.projection_heads,
                hidden_size=self.config.hidden_size,
                k=self.config.moe_k,
                noisy_gating=True,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
        elif self.config.projection_layers == 100:
            self.projection_layer = ProjectionMLP(self.config.hidden_size)
        elif self.config.projection_layers > 0:
            self.projection_layer = MLPLayer(
                self.config.hidden_size, self.config.projection_layers, self.config.projection_heads
            )
        else:
            self.projection_layer = None

        if self.config.do_mlm:
            self.lm_head = BertLMPredictionHead(self.config)

        self.prediction_layer = None

        if self.config.discriminate:
            self.discriminator = AugmentationDiscriminator(
                input_size=self.config.hidden_size * 2,
                num_layers=self.config.discriminator_layers,
                num_labels=2
                if self.config.discriminate_original or self.config.discriminate_order
                else (1 + self.config.num_augmentations),
                gradient_reverse_multiplier=self.config.gradient_reverse_multiplier,
                dropout=self.config.discriminator_dropout,
            )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        eval=False,
        mlm_input_ids=None,
        mlm_labels=None,
    ):
        loss = torch.tensor(0.0, device=self.transformer_model.device)
        # MLM auxiliary objective
        if mlm_input_ids is not None and mlm_labels is not None:
            bs, num_sent, _ = mlm_input_ids.shape
            if num_sent <= 3:  # single sentence
                nl_idx = 0
            elif num_sent < 6:  # pair sentence, with one aug at a time
                nl_idx = 2
            else:  # pair sentence, positive and negative aug
                nl_idx = 3

            # mlm_input_ids = mlm_input_ids.view((-1, mlm_input_ids.size(-1)))
            mlm_outputs = self.transformer_model(
                mlm_input_ids[:, nl_idx],
                attention_mask=attention_mask.view(bs, num_sent, -1)[:, nl_idx],  # on aug adds this in batch
                token_type_ids=token_type_ids.view(bs, num_sent, -1)[:, nl_idx],
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
            # mlm_labels = mlm_labels.view(-1, mlm_labels.size(-1))
            prediction_scores = self.lm_head(mlm_outputs.hidden_states[-1])
            masked_lm_loss = nn.CrossEntropyLoss()(
                prediction_scores.view(-1, self.config.vocab_size), mlm_labels[:, nl_idx].reshape(-1)
            )
            loss += self.config.mlm_weight * masked_lm_loss

        outputs = self.transformer_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(attention_mask, outputs)
        if self.projection_layer:
            pooler_output = self.projection_layer(pooler_output, eval=eval)
            if isinstance(self.projection_layer, MoE):
                pooler_output, moe_loss = pooler_output
                loss += moe_loss

        if self.prediction_layer and not eval:
            prediction_output = self.prediction_layer(pooler_output)
            pooler_output = pooler_output.detach()

        # TODO for actual classification tasks, we need to train classifier head on top of
        # TODO projection outputs
        # logits, loss = self.classifier(pooler_output)

        # append pooled and projected output to hidden states
        if self.prediction_layer and not eval:
            outputs.hidden_states = outputs.hidden_states + (pooler_output, prediction_output)
        else:
            outputs.hidden_states = outputs.hidden_states + (pooler_output,)

        if not return_dict:
            return ((outputs.loss,) + outputs[2:]) if outputs.loss is not None else outputs[2:]

        return SequenceClassifierOutput(
            loss=loss,
            logits=outputs.logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
