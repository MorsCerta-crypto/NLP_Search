import numpy as np

from n_gram.corpus import Corpus
from n_gram.model import NgramModel


def eval(learn_up_to: int, corpus: Corpus, test: bool = False) -> NgramModel:
    """evaluate the training of a model, for training on [2 : learn_up_to] N-Grams"""
    model = NgramModel()
    model.learn(corpus, learn_up_to=learn_up_to)

    # print('Training cross ent: {} (count={})'.format(*cross_ent(model, corpus, learn_up_to, test=False)))
    if test:
        print('Test cross ent: {} (count={})'.format(*cross_ent(model, corpus, learn_up_to, test=test)))

    return model


def cross_ent(model: NgramModel, corpus: Corpus, learn_up_to: int, test: bool):
    """calculate cross entropy for trianed model"""
    cross_ent = 0.0
    count = 0
    pred = None
    distr = None
    for ngram in corpus.get_ngrams(learn_up_to, test=test):
        context = ngram[0:learn_up_to-1]
        pred = ngram[learn_up_to-1]
        distr = model.predict(context)

        # only count ngrams that occurred in the training data
        if pred in distr:
            cross_ent -= np.log2(distr[pred])
            count += 1
    if count != 0:
        cross_ent /= count
    else:
        raise ValueError(f"count is zero, {pred=} is not in {distr=}")
    return cross_ent, count


def load_and_train(cleaned_text: list[list[str]], learn_up_to: int, test: bool = False) -> NgramModel:
    """loads a corpus with cleaned text[str] and trains a ngram model"""

    corpus = Corpus(cleaned_text)
    model = eval(learn_up_to, corpus, test)
    return model


def main():
    text: str = "Darüber hinaus erfordert eine Rüge der Verletzung des Anspruchs auf Gewährung rechtlichen Gehörs regelmäßig die substantiierte Darlegung dessen, was die Prozesspartei bei ausreichender Gewährung rechtlichen Gehörs noch vorgetragen hätte und inwiefern dieser Vortrag zur Klärung des geltend gemachten Anspruchs geeignet gewesen wäre. 10StRspr. vgl. BVerwG, Beschluss vom 31.8.2016 ‒ 4 B 36.16 ‒, juris, Rn. 3. 11An einer solchen substantiierten Darlegung fehlt es. Anhaltspunkte dafür, dass dem Kläger eine weitere Darlegung nicht möglich war, bestehen nicht. Abgesehen davon, dass er auch zur Begründung seines Zulassungsantrags hätt"
    corpus = Corpus(text)
    eval(2, corpus)
    eval(3, corpus)
    eval(5, corpus)
    eval(10, corpus)


if __name__ == '__main__':
    main()
