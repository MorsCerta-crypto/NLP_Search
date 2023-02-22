

import collections
from typing import Union
import numpy as np
import spacy

np.random.seed(0)


class Corpus:
    """this class provides words, sentences and n-grams for training a model
    """

    def __init__(self, text: Union[str, list[list[str]]], test_percentage: float = 0.1):
        self.test_percentage = test_percentage

        # use spacy NLP to do the tokenization and sentence boundary detection
        nlp = spacy.load('de_core_news_lg')
        if isinstance(text, str):
            self.is_list_format = False
            self.doc = nlp(text)

        elif isinstance(text, list):
            self.is_list_format = True
            self.sentences = text

        else:
            raise TypeError("given text must be a list or a str")

    def get_words(self):
        """provide words from self.sentences"""
        if not self.is_list_format:
            for token in self.doc:
                yield token.text
        else:
            for sent in self.sentences:
                for word in sent:
                    yield word

    def get_sentences(self, test: bool = False):
        """provide sentences from the document"""
        if not self.is_list_format:
            for sent in self.doc.sents:
                # split into training and test sentences, according to the given percentage
                if (np.random.random() >= self.test_percentage and not test) or \
                        (np.random.random() < self.test_percentage and test):
                    yield sent
        else:
            for sent in self.sentences:
                if (np.random.random() >= self.test_percentage and not test) or \
                        (np.random.random() < self.test_percentage and test):
                    yield sent

    def get_ngrams(self, n: int, test: bool = False):
        """divide sentences into slices of length n"""
        for sent in self.get_sentences(test=test):

            if len(sent) < 10:
                continue
            for pos in range(len(sent)):

                if len(sent)-pos < n:
                    break
                if self.is_list_format:
                    yield (*[sent[pos+i] for i in range(n)],)
                else:
                    yield (*[sent[pos+i].text for i in range(n)],)


def print_most_common(n: int, corpus):
    counter = collections.Counter(corpus.get_ngrams(n))
    print('\nThe most common {}-grams:'.format(n))
    for k, v in counter.most_common(5):
        print('{}: {}'.format(k, v))


def main():
    text = "Der Antrag des Klägers auf Zulassung der Berufung gegen das auf die mündliche Verhandlung"
    corpus = Corpus(text)

    print('Number of words in corpus: ', len(list(corpus.get_words())))
    print('Number of training sentences in corpus: ', len(list(corpus.get_sentences())))
    print('Number of test sentences in corpus: ', len(list(corpus.get_sentences(test=True))))
    print('Size of alphabet:', len(set(corpus.get_words())))

    print_most_common(1, corpus)
    print_most_common(3, corpus)
    print_most_common(5, corpus)


if __name__ == "__main__":
    main()
