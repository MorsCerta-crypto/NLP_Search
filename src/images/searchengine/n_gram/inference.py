import ast
import time

from Parser.parser import Quintuple
from preprocessing.preprocessor import Preprocessor
from WordEmbeddings.create_word_embeddings import WordEmbeddingsInference, DocVec
from WordEmbeddings.lexsub import LexSub
from n_gram import train_input_prediction
from n_gram.model import NgramModel
import pandas as pd


class Inference:
    """this class combines the use of all models for input predictions:
    ngram-model, wordembedding - model, docvec-model, lexsub-model"""

    def __init__(self):
        self.word_embedding_model = WordEmbeddingsInference(phrase=False)  # use phrase model or normal single-word model
        self.n_gram_model = NgramModel()
        self.n_gram_model.load_params()
        self.doc_model = DocVec(train=False)
        self.substitution = LexSub(n_candidates=10)

    def get_lexical_substitution(self, pos: int, context: list[str]):
        """find a lexical substitution for a given context"""
        return self.substitution.lex_sub(word_POS=pos, sentence=context)

    def predict_on_context_word_embeddings(self, context: list[str], top_n=5):
        """ load a trained model and predict a string on a context"""
        return self.word_embedding_model.get_most_similar(context[-1], top_n=top_n)

    def predict_on_context_n_grams(self, context: list[str]):
        """
        use a n_gram_model to predict an alternative for
        the last word and suggestions for the next word
        """
        return self.n_gram_model.predict_next_word(context)

    def predict_similar_sentence(self, context: list[str]):
        """ uses a Doc2Vec model to find the most similar sentence in the dataset"""
        return self.doc_model.most_similar(context)

    def search_quintuple(self, quintuple: Quintuple, list_of_quintuples: list[str]) -> list[dict[str, str]]:
        """looks up the most similar uintuple in a list of str representations of quintuples

        Args:
            quintuple (Quintuple): quintuple to find a similar quintuple to
            list_of_quintuples (list[str]): string representation of a quintuple(same as str repr of dict)]

        Returns:
            List[Quintuple]: full quintuples that are the most similar to given quintuple
        """
        # count matching keys for search_quintuple in quintuples
        sim_count_map = []
        search_dict = quintuple.serialize()
        same_values = None

        for entry in list_of_quintuples:
            try:
                dbquintuple = ast.literal_eval(entry)
            except Exception:
                print("entry", entry)
                print("Exception parsing a string in inference")
                continue
            for key, value in dbquintuple.items():
                same_values = 0
                for k, val in search_dict.items():
                    if key == k:
                        same_values += len(intersection(val, value))

            sim_count_map.append(same_values)

        max_score = max(sim_count_map)
        print("maximum score achieved: ", max_score)
        indices = sorted(range(len(sim_count_map)), key=lambda i: sim_count_map[i], reverse=True)[:2]
        elems = (list_of_quintuples[e] for e in indices)
        top_matches: list[dict[str, str]] = [ast.literal_eval(elem) for elem in elems]
        return top_matches


def intersection(lst1, lst2):
    """calculate same elemts in two lists"""
    return [value for value in lst1 if value in lst2]


def main(preprocess: bool = False, do_training: bool = False):

    p = Preprocessor()

    if preprocess:
        p.preprocess()

    # get input (search term)
    sentence = ['das', 'Verfahren', 'der', '9.', 'Zivilkammer']
    sentence2 = ['Abgabe', 'an', 'das', 'im']
    sentence3 = ['das', 'verfahren', 'der', '9.', 'end']
    sentence4 = ['Klage', 'und', 'Eingang', 'von', 'Klageerwiderung', 'und']
    sentence5 = ['Verfahren', 'der', '9.', 'Ziv']

    sentences = [sentence, sentence2, sentence3, sentence4, sentence5]

    # read quintuples from csv/Database
    path = "results_of_parsing.csv"
    df = pd.read_csv(path)
    list_of_quintuples = df["quintuples"].tolist()
    train_sentences = df["sentence"]
    if do_training:
        train_input_prediction.run_training(learn_up_to=8, sentences=train_sentences)

    # make quintuple from input sentence
    quintuple = p.preprocess_sentence(" ".join(sentence))

    i = Inference()

    start_time = time.time()
    # search
    for sentence in sentences:
        try:
            full_quintuple = i.search_quintuple(quintuple, list_of_quintuples)
            n_grams = i.predict_on_context_n_grams(context=sentence)
            embeddings = i.predict_on_context_word_embeddings(context=sentence)
            doc_sim_sentnece = i.predict_similar_sentence(sentence)
            lex_sub = i.get_lexical_substitution(context=sentence, pos=f"{sentence[-1]}.n")

            print("full quintuples that match: ", full_quintuple)
            print("NGramModel predicted: ", n_grams, "\nWordEmbeddings predicted: ", embeddings)
            print("doc_sim_sentnece: ", doc_sim_sentnece)
            print("lexical sub: ", lex_sub)
        except ArithmeticError:
            continue
    stop_time = time.time()
    print("Time elapsed for inference: ", stop_time - start_time)


if __name__ == '__main__':
    main(preprocess=False, do_training=False)
