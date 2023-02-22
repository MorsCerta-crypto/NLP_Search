
import os
import pickle
from collections import Counter
from typing import Optional

import pandas as pd
import spacy
from Utilities.read_config import read_config

from n_gram.corpus import Corpus

# https://github.com/ad-freiburg/qac/blob/master/qac.py
# https: // github.com/nltk/nltk/tree/model/nltk/model
# https: // ad-publications.cs.uni-freiburg.de/theses/Bachelor_Natalie_Prange_2016.pdf


class NgramModel:
    ngrams: Counter
    alphabet: set[str]

    def __init__(self, ngrams: Optional[Counter] = None, alphabet: Optional[set[str]] = None, params_file: str = "n_gram/data/data"):
        self.max_n = None
        self.params_file = params_file
        self.nlp = spacy.load('de_core_news_lg')

        # take given ngrams or load them from file#
        self.ngrams = ngrams

        self.alphabet = alphabet
        # determine max n for predictions

    def load_params(self):
        """ load params for inference"""
        self.alphabet = self.load_alphabet()
        print("loaded alphabet of len", len(list(self.alphabet)))
        self.ngrams = self.load_ngrams()
        print("loading ngrams of length %d" % len(list(self.ngrams)))
        self.calculate_max_n()

    def learn(self, corpus: Corpus, learn_up_to: int):
        """trainin: fits ngrams and alphabet"""
        #print("corpus text:", corpus.get_words())
        self.ngrams = None
        if learn_up_to > 1:
            for n in range(1, learn_up_to):
                if not self.ngrams:
                    self.ngrams = Counter(corpus.get_ngrams(learn_up_to))
                else:
                    self.ngrams += Counter(corpus.get_ngrams(n))

        self.save_ngrams()
        self.alphabet: set[str] = set(corpus.get_words())
        self.save_alphabet()
        self.calculate_max_n()

    def calculate_max_n(self):
        # there are no ngrams stored for prediction above this length
        lengths = set(len(elem) for elem in dict(self.ngrams).keys())
        self.max_n = max(lengths)
        assert lengths == set(range(2, self.max_n+1))

    def predict(self, context: list[str]) -> dict[str, float]:
        """uses ngrams to predict a following word to context

        Args:
            context (list[str]): [information about the current sentence as a tokenized sentence]

        Raises:
            ValueError: [context is to short]

        Returns:
            dict[str, float]: [predicted following words]
        """
        # load ngrams for context prediction
        if self.max_n is None:
            self.calculate_max_n()

        if len(context) >= self.max_n:
            context = context[-self.max_n+1:]

        matches: dict[str, int] = {}
        for word in self.alphabet:
            count = self.ngrams[tuple(context) + (word,)]
            if count > 0:
                matches[word] = count

        total_count: float = sum(matches.values(), 0.0)
        return {k: v / total_count for k, v in matches.items()}

    def predict_str(self, context_str: str):
        """predict a string from a given context"""
        context: list[str] = [token.text for token in self.nlp(context_str)]
        return self.predict(context)

    def check_if_word_is_in_alphabet(self, in_word):
        if in_word in self.alphabet:
            return True
        if in_word.lower() in self.alphabet:
            return True
        print(f"{in_word} is not in alphabet")
        return False

    def predict_next_word(self, context: list[str]):
        """return most likely predictions
        concidered were 4 cases:
        1. last word is in alphabet -> predict next word
        2. last word is in alphabet but only because its part of another word -> predict corrected last word
        3. last word is not in alphabet and should be replaced -> predict corrected last word and next word
        4. last word is not in alphabet but part of a word in alphabet -> predict corrected last word an next word

        returns possible_endings[list[list[str]]], where first element is a list of possible last words with their next word and second element is a list of possible next words
        """
        # possible_endings[0] : pairs of corrected last words with matching next words
        # possible_endings[1] : predicted next words for current last word

        possible_endings: list[list[str]] = []
        corr_last_words: list[str] = []
        last_word_and_next_word = []
        next_words: list[str] = []
        # word is not in alphabet
        if not self.check_if_word_is_in_alphabet(context[-1]):
            # try to find other word instead of last word in context
            corr_last_words = list(self.predict(context[:-1]).keys())

            # try all possible endings without actual ending
            last_word_and_next_word = []
            for word in corr_last_words:
                new_sentence = context
                new_sentence[-1] = word
                for next_word in self.predict(new_sentence).keys():
                    last_word_and_next_word.append([word, next_word])
        else:
            # predict only the following word
            next_words = list(self.predict(context).keys())

        # set a maximum of 10 suggestions
        if len(last_word_and_next_word) > 9:
            last_word_and_next_word = last_word_and_next_word[:10]
        if len(next_words) > 9:
            next_words = next_words[:10]
        possible_endings.append(last_word_and_next_word)
        possible_endings.append(next_words)

        return possible_endings

    def check_matches_for_characters(self, chars, predictions):
        """ when ngrams predict a new word but the user already has give a character or more of the
        next word, we should check if any of the results has these characters in its beginning"""
        for prediction in predictions:
            # one of the predictions has a common beginning with the userinput
            if prediction.startswith(chars):
                return prediction
            # user input does not match the predictions
            else:
                return "<unk>"

    def check_for_file(self, file):
        split_path = os.path.split(file)
        if not os.path.exists(split_path[0]):
            os.makedirs(split_path[0])
        if not os.path.exists(file):
            with open(file, "w"):
                pass

    def save_ngrams(self, n=None) -> None:
        """store parameters in a file"""
        print(f"save n_grams to file {self.params_file}_ngrams")
        file = f"{self.params_file}_ngrams.txt"
        self.check_for_file(file)

        with open(file, "wb+") as f:
            pickle.dump(self.ngrams, f)

    def save_alphabet(self) -> None:
        file = f"{self.params_file}_alphabet.txt"
        self.check_for_file(file)

        with open(file, "wb+") as f:
            pickle.dump(self.alphabet, f)

    def load_alphabet(self) -> set[str]:
        file = f"{self.params_file}_alphabet.txt"
        if not os.path.isfile(file):
            print("creating alphabet file")
            with open(file, "wb+") as f:
                pickle.dump(set("<unk>"), f)
        with open(f"{self.params_file}_alphabet.txt", "rb") as f:
            data = pickle.load(f)
            return data

    def load_ngrams(self) -> Counter:
        """loads ngrams from file"""
        file = f"{self.params_file}_ngrams.txt"
        if not os.path.isfile(file):
            print("creating ngrams file")
            with open(file, "wb+") as f:
                pickle.dump(Counter("<unk>"), f)
        with open(f"{self.params_file}_ngrams.txt", "rb") as f:
            return pickle.load(f)


def load_parsed_data():
    file = read_config()
    df = pd.read_csv(file)
    text = df["sentence"].tolist()
    # a sentence needs to be list[str]
    # so sentences should be a list[list[str]]
    sentences = [s.split() for s in text]
    print(sentences[0])
    return sentences


if __name__ == "__main__":
    text = load_parsed_data()
    corpus = Corpus(text)

    model = NgramModel()
    model.learn(corpus, learn_up_to=4)
    pmodel = NgramModel()
    
    inp = 'als funktional zuständig'
    ans = model.predict_next_word(inp)
    
    print(f"Model prediction: {ans} for input: {inp}")
