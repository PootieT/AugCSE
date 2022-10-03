import torch
import os
from transformers import AutoTokenizer, AutoModel
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Experimenter:
    def __init__(self):
        pass

    def load_model(self, model_name):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device), tokenizer

    def experiment_logger(self):
        pass

    def mean_pooling(sefl, model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def predict_sentence_embeddings(self, sentences, model, tokenizer, batch_size=16):
        # Tokenize sentences
        for i in range(0, len(sentences), batch_size):
            encoded_input = tokenizer(sentences[i : i + batch_size], padding=True, truncation=True, return_tensors="pt")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            encoded_input = encoded_input.to(device)
            # Compute token embeddings
            with torch.no_grad():
                model_output = model(**encoded_input)
            # Perform pooling. In this case, mean pooling.
            batch_embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
            if i == 0:
                sentence_embeddings = batch_embeddings.cpu().detach().numpy()
            else:
                sentence_embeddings = np.append(sentence_embeddings, batch_embeddings.cpu().detach().numpy(), axis=0)
        return sentence_embeddings

    def predict_sentence_embeddings_with_batcher(self, sentences, batcher_func, batch_size=16):
        # Tokenize sentences
        for i in range(0, len(sentences), batch_size):
            batch_embeddings = batcher_func({}, sentences[i : i + batch_size])
            if i == 0:
                sentence_embeddings = batch_embeddings.cpu().detach().numpy()
            else:
                sentence_embeddings = np.append(sentence_embeddings, batch_embeddings.cpu().detach().numpy(), axis=0)
        return sentence_embeddings

    def train_single_layer(self, X_train, y_train, eval_type="linear"):
        if eval_type == "linear":
            clf = LogisticRegression(solver="lbfgs", multi_class="multinomial", max_iter=1000)
        elif eval_type == "mlp":
            clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)
        elif eval_type == "svm":
            clf = make_pipeline(StandardScaler(), SVC(kernel="linear", probability=True))
        else:
            raise ValueError("Layer type not recognized it should be either linear, mlp or svm")
        clf.fit(X_train, y_train)
        return clf

    def pre_experiment_logger():
        pass
