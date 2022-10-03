from testing_part.src.experimenters.experimenter import Experimenter
from sklearn.metrics import accuracy_score


class SentExperimenter(Experimenter):
    def __init__(self):
        pass

    def mnli_experiment(self, model, tokenizer, nli_dataset):
        refs, pos_samples, neg_samples = nli_dataset.preprocess_pair()
        print(f"Length of pairs is {len(refs)}")
        ref_embeddings = self.predict_sentence_embeddings(refs, model, tokenizer)
        pos_embeddings = self.predict_sentence_embeddings(pos_samples, model, tokenizer)
        neg_embeddings = self.predict_sentence_embeddings(neg_samples, model, tokenizer)
        count = 0
        for i in range(len(ref_embeddings)):
            pos_score = ref_embeddings[i].dot(pos_embeddings[i])
            neg_score = ref_embeddings[i].dot(neg_embeddings[i])
            if pos_score > neg_score:
                count += 1
        print(f"Accuracy is {count/pos_embeddings.shape[0]*100}%")

    def mnli_experiment_with_batcher(self, nli_dataset, batcher_func, batch_size=16):
        refs, pos_samples, neg_samples = nli_dataset.preprocess_pair()
        print(f"Length of pairs is {len(refs)}")
        ref_embeddings = self.predict_sentence_embeddings_with_batcher(refs, batcher_func, batch_size)
        pos_embeddings = self.predict_sentence_embeddings_with_batcher(pos_samples, batcher_func, batch_size)
        neg_embeddings = self.predict_sentence_embeddings_with_batcher(neg_samples, batcher_func, batch_size)
        count = 0
        for i in range(len(ref_embeddings)):
            pos_score = ref_embeddings[i].dot(pos_embeddings[i])
            neg_score = ref_embeddings[i].dot(neg_embeddings[i])
            if pos_score > neg_score:
                count += 1
        acc = count / pos_embeddings.shape[0] * 100
        print(f"Accuracy is {count/pos_embeddings.shape[0]*100}%")
        return acc

    def cola_experiment(self, model, tokenizer, cls_train_dataset, cls_val_dataset, eval_type="linear"):
        print(f"Length of train data is {len(cls_train_dataset.dataset)}")
        print(f"Length of validation data is {len(cls_val_dataset.dataset)}")
        X_train = self.predict_sentence_embeddings(cls_train_dataset.dataset["sentence"], model, tokenizer)
        X_val = self.predict_sentence_embeddings(cls_val_dataset.dataset["sentence"], model, tokenizer)
        y_train = cls_train_dataset.dataset["label"]
        y_val = cls_val_dataset.dataset["label"]
        clf = self.train_single_layer(X_train, y_train, eval_type=eval_type)
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Linear accuracy is {accuracy}%")

    def cola_experiment_with_batcher(
        self, cls_train_dataset, cls_val_dataset, batcher_func, batch_size=16, eval_type="linear"
    ):
        print(f"Length of train data is {len(cls_train_dataset.dataset)}")
        print(f"Length of validation data is {len(cls_val_dataset.dataset)}")
        X_train = self.predict_sentence_embeddings_with_batcher(
            cls_train_dataset.dataset["sentence"], batcher_func, batch_size
        )
        X_val = self.predict_sentence_embeddings_with_batcher(
            cls_val_dataset.dataset["sentence"], batcher_func, batch_size
        )
        y_train = cls_train_dataset.dataset["label"]
        y_val = cls_val_dataset.dataset["label"]
        clf = self.train_single_layer(X_train, y_train, eval_type=eval_type)
        y_pred = clf.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Linear accuracy is {accuracy}%")
        return accuracy

    def pi_experiment(self, model, tokenizer, pi_dataset):
        pass
