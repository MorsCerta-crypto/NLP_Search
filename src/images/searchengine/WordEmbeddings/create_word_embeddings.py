
from gensim.models import Phrases, Word2Vec
from typing import Optional
from nltk.corpus import stopwords
from tqdm import tqdm
from typing import Optional
import time
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from Utilities.read_config import read_config

# http://ceur-ws.org/Vol-2410/paper7.pdf
# https://ad-publications.cs.uni-freiburg.de/theses/Bachelor_Natalie_Prange_2016_presentation.pdf

# skipgram: https://www.geeksforgeeks.org/implement-your-own-word2vecskip-gram-model-in-python/


class WordEmbeddingsInference:
    """This class combines all methods for loading and using a word embeddings model"""

    def __init__(self, model: Word2Vec = None, path: str = "", phrase: bool = False):
        self.config = read_config()
        self.model_path = path
        if model:
            self.model = model
        elif phrase:
            self.model = self.load_phrase_model(self.config["phrase_model_file"])
        else:
            self.model = self.load_trained_model(self.config["model_file"])

    def load_trained_model(self, name: str):
        """load a Word2Vec Model"""
        print("loading word embeddings model from file ...")
        return Word2Vec.load(self.model_path + name)

    def load_phrase_model(self, name: str):
        return Phrases.load(self.model_path + name)

    def n_sim(self, s1, s2):
        """calculates the similarity of two sentences"""
        return self.model.wv.n_similarity(s1, s2)

    def wm_distance(self, s1: list[str], s2: list[str]):
        """calculate wm-distance between two sentences"""
        distance = self.model.wmdistance(s1, s2)
        return distance

    def get_most_similar(self, word: str, top_n: int = 10) -> Optional[list[str]]:
        """return the top_n exact neighbor words for a given word"""

        try:
            vec = self.model.wv[word]
            ans = [words for words, probs in self.model.wv.most_similar([vec], top_n) if probs > 0.]

            print("text: ", word, " most similar: ", ans)
            return ans
        except KeyError:
            print(type(word), word)
            print("Key not available in model.wv!")

    @property
    def get_model_words(self):
        """return the words of a model"""

        if not isinstance(self.model, Word2Vec):
            raise TypeError("model must be a Word2Vec")

        # summarize vocabulary
        words = list(self.model.wv.vocab)
        return words


class DocVec:
    """This class helps to train a Doc2Vec model from a given corpus and
    can find similar sentences from the corpus to a given sentence"""

    model: Doc2Vec

    def __init__(self, model: Doc2Vec = None, train: bool = True, path: str = ""):
        self.config = read_config()
        self.stopwords = stopwords.words("german")
        self.tagged_documents = []
        self.model = None

        # load model from file
        self.model_path = path
        if not model and not train:
            self.model = self.load_model()
        elif model:
            self.model = model

    def train_model(self, documents: list[list[str]], labels: list[str]):
        """trains and saves a Doc2Vec model with documents and labels

        Args:
            documents (list[list[str]]): a list of sentences, where each sentences is tokenized into words
            labels (list[str]): a unique label for a sentence
        """

        # make sure there are as many sentences as labels
        assert len(labels) == len(documents)

        # split sentences into lists of words
        tokenized_sentences = [self.remove_stopwords(x) for x in documents]
        # model only accepts data in form of a TaggedDocument
        self.tagged_documents = [TaggedDocument(doc, tags=[label]) for label, doc in zip(labels, tokenized_sentences)]

        self.model = Doc2Vec(vector_size=100, min_count=2, epochs=40, workers=8)
        self.model.build_vocab(self.tagged_documents)

        start = time.time()
        self.model.train(self.tagged_documents, total_examples=self.model.corpus_count, epochs=self.model.epochs)
        end = time.time()
        print(f"Time elapsed for training: {end-start:.2f}")

        self.save_model()

    def load_model(self):
        return Doc2Vec.load(self.model_path+self.config["embeddings_file"])

    def save_model(self):
        self.model.save(self.model_path+self.config["embeddings_file"])

    def most_similar(self, sentence) -> list[str]:
        """returns labels of sentences that are similar to the input"""
        assert self.model is not None
        vector = self.model.infer_vector(self.remove_stopwords(sentence))
        refs = self.model.docvecs.most_similar(positive=[vector], topn=3)
        print("refs:", refs)
        prob_pref = [ref for ref, prob in refs if prob > 0.3]
        return prob_pref

    def remove_stopwords(self, x):
        return [w.lower() for w in x if not w in self.stopwords]


class WordEmbeddingsTraining:
    """"this class combines all methods for training a word embedding model with sentences"""

    def __init__(self, path: str = '', save: bool = False):
        #split_path = os.path.split(model_path)[0]
        # if not os.path.exists(split_path):
        #     os.makedirs(split_path)
        self.model_path = path
        self.save = save

    def train(self, sentences) -> Word2Vec:
        """trains a word embeddings model and saves params"""
        self.model = self.create_word_embeddings(sentences)
        # self.train_phrase_model(sentences)

        self.save_model("model.bin")
        return self.model

    def create_word_embeddings(self, sentences: list[list[str]]) -> Word2Vec:
        """train a Word2Vec Model
        sentences must be iterable and restartable not just a generator"""
        # train model
        start = time.time()
        model = Word2Vec(sentences, vector_size=100, min_count=2, workers=8, epochs=30, window=3, sg=0)
        end = time.time()
        print(f"Time elapsed for training: {end-start:.2f}")
        return model

    def train_phrase_model(self, sentences: list[str], path="phrase.model"):
        """train a model that can handle phrases not just single words"""
        bigram_transformer = Phrases(sentences)
        self.model = Word2Vec(bigram_transformer[sentences], min_count=1)
        if self.save:
            self.save_model(path)

    def save_model(self, filename):
        """save Word2Vec model"""
        self.model.save(self.model_path+filename)
        print(f'save model to {self.model_path+filename}')


def main():
    text = ["Das ist der erste Satz.", "Diesem Satz folgt ein Zweiter."]
    sentences = [s.split(" ") for s in text]
    train_phrase_model = WordEmbeddingsTraining()
    train_phrase_model.train(sentences)

    inference_phrase_model = WordEmbeddingsInference()
    ans = inference_phrase_model.get_most_similar("folgt", 2)
    print(f"Model answered with: {ans}")


if __name__ == "__main__":
    main()
