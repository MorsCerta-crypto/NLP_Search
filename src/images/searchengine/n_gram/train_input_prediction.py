"""
    1. Parsed Daten laden aus der Datenbank
    2. Word Embeddings erzeugen
    3. ngrams trainieren
    """

from n_gram.model import NgramModel
from WordEmbeddings.create_word_embeddings import WordEmbeddingsTraining
from Utilities.read_config import read_config
from MongoDBController.mongodb_controller import MongoDBController
from typing import Optional, List, Tuple, Any
from WordEmbeddings.create_word_embeddings import WordEmbeddingsTraining, DocVec
from n_gram import training
import time
from typing import Any, List, Optional, Tuple

import pandas as pd


class Training:
    sentences: list[list[str]]

    def __init__(self, word_embedding_training: WordEmbeddingsTraining, doc_embedding_training: DocVec, from_db: bool = True) -> None:

        self.word_embedding_training = word_embedding_training
        self.doc_embedding_training = doc_embedding_training
        self.from_db = from_db
        if self.from_db:
            self.controller = MongoDBController(collection="CourtDecisions")
        self.sentences = []

    def train_n_grams_and_word_embeddings(self, sentences: Optional[List[List[str]]], labels: List[str], learn_up_to: int = 5) -> Tuple[NgramModel, Any]:
        """combines training of ngrams and wordembeddings"""
        if sentences:
            self.sentences = sentences
        else:
            if not self.from_db and self.controller is None:
                raise ValueError("either connection or senteneces required")
        embedding_model = self.make_word_embeddings()
        doc_embedding = self.make_doc_embeddings(labels)
        ngram_model = self.train_n_grams(learn_up_to)

        return ngram_model, embedding_model, doc_embedding

    def load_parsed_data_from_db(self, num_sentences: int = 10000):
        """load sentences from database"""
        self.controller.connect()
        text = self.controller.get_sentences(num_sentences)
        labels = self.controller.get_labels(num_sentences)
        self.controller.disconnect()
        return text, labels

    def make_word_embeddings(self):
        """trains word embedding model"""
        return self.word_embedding_training.train(self.sentences)

    def make_doc_embeddings(self, labels):
        """trains Doc2Vec Model"""
        return self.doc_embedding_training.train_model(self.sentences, labels)

    def train_n_grams(self, learn_up_to: int) -> NgramModel:
        """trains a ngram model and stores parameters in a dictionary"""
        return training.load_and_train(cleaned_text=self.sentences, learn_up_to=learn_up_to)

    def load_parsed_data(self, num_sentences: int = 10000):
        """returns the data needed for training"""

        if self.from_db:
            text, labels = self.load_parsed_data_from_db(num_sentences)
        else:
            file = read_config()["parsing_results_file"]
            df = pd.read_csv(file)
            text = df["sentence"].tolist()
            labels = df["id"].tolist()
        # a sentence needs to be list[str]
        # so sentences should be a list[list[str]]
        sentences = [s.split() for s in text]

        for label in labels:
            if label in [8, "L", "E", "_", 2, 1]:
                raise ValueError("LABELS ERROR")
        # print(sentences[:5])
        return sentences, labels


def run_training(learn_up_to, from_db: bool = False, num_sentences: int = 10000):
    start_time = time.time()
    we = WordEmbeddingsTraining()
    do = DocVec(train=True)
    Trainer = Training(we, do, from_db=from_db)
    sentences, labels = Trainer.load_parsed_data(num_sentences)
    ngrams = Trainer.train_n_grams_and_word_embeddings(sentences, labels=labels, learn_up_to=learn_up_to)
    stop_time = time.time()
    print("Time elapsed for full training: ", stop_time - start_time)
    return ngrams


if __name__ == "__main__":
    models = run_training(learn_up_to=6, from_db=False, num_sentences=10000)
