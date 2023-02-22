from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from Utilities.read_config import read_config


class SentTokenizer:
    """This class implements methods for splitting a text into sentences and words"""

    def __init__(self) -> None:
        self.abbreviations = ["az", "bzw", "i.h.v", "i.s.d", "i.v.m", "usw", "v.a", "z.b", "etc", "zzgl", "z.t", "z.z", "z.zt", "abs",
                              "zur", "zus", "z.hd", "v.t", "v.u.z", "vgl", "u.ä", "u.a", "tel", "s.u", "s.o", "o.g", "mwst", "aufl",
                              "max", "min", "i.a", "inh", "nr", "az", "evtl", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0", "rn"]
        self.tokens: list[str] = []
        self.config = read_config()
        self.abbreviations_file = self.config["abbreviations_file"]
        self.__fetch_abbreviations_list()

    def get_tokens(self) -> list[str]:
        """return tokens of the class"""
        return self.tokens

    def tokenize(self, text: str) -> list[str]:
        """use nltk to tokenize a sentence; returns a list of the words from the sentence"""
        self.__nltk_tokenize_text_to_sent(text)
        return self.get_tokens()

    def __nltk_tokenize_text_to_sent(self, text: str) -> None:
        """uses nltk to devide a text into sentences; it regards a given list of german abbreviations"""
        punkt_param = PunktParameters()
        punkt_param.abbrev_types = set(self.abbreviations)
        self.tokens = PunktSentenceTokenizer(punkt_param).tokenize(text)
        self.__handle_split_by_number()

    def __handle_split_by_number(self):
        """
        make sure text is not split
        after a Number followed by a dot as in '22.01.2010'
        or in listings like '1.point one 2. point two'
        """
        idx = 0
        while (idx < len(self.tokens)):
            if self.tokens[idx][-1] == "." and self.tokens[idx][-2].isnumeric():
                token = self.tokens.pop(idx)  # Reduce Array Size!
                idx -= 1
                self.tokens[idx] = token + " " + self.tokens[idx]

            idx += 1

    def __fetch_abbreviations_list(self):
        """read file with abbreviations"""

        with open(self.abbreviations_file, "r", encoding="utf-8") as f:
            content = f.readlines()
        for line in content:
            line = line[:-1]  # Remove \n
            line = line if line[-1] != "." else line[:-1]
            self.abbreviations.append(line.lower())
        # print(f"{self.abbreviations=}")
