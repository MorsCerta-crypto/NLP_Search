# this code comes from : https://github.com/Mchristos/lexsub/

import re
from nltk.corpus import stopwords

stopw = stopwords.words("german")


def get_words(s):
    """ Extract a list of words from a sentence string with punctuation, spaces etc
    s = sentence
    """
    s = " ".join(s)
    # strip punctuation
    s = re.sub(r'[^\w\s]', '', s)
    # replace newline
    s = s.replace('\n', ' ')
    # get rid of spaces
    s = " ".join(s.split())
    return s.split(' ')


def unique(iter):
    "removes duplicates from iterable preserving order"
    result = list()
    seen = set()
    for x in iter:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def process_candidates(candidates, target):
    """ words to lower case, replace underscores, remove duplicated words,
        filter out target word and stop words """

    filterwords = stopw + [target]
    return unique(filter(lambda x: x not in filterwords,
                  map(lambda s: s.replace('_', ' '), candidates)))
