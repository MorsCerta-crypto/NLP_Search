import json
from dataclasses import dataclass
from typing import List

import pandas as pd
from Utilities.read_config import read_config
from MongoDBController.mongodb_controller import MongoDBController
from Parser.parser import Quintuple, QuintupleReducer
from SentTokenizer.sent_tokenizer import SentTokenizer
from TextCleaner.cleaner import clean


@dataclass
class Preprocessor:

    def __post_init__(self):
        self.config: dict[str, str] = read_config()
        self.source_path: str = self.config["database_file"]
        self.content: List[str] = []
        self.mongoDB = MongoDBController()
        self.mongoDB.connect()

        #self.TC = TextCleaner()
        self.ST = SentTokenizer()
        self.QP = QuintupleReducer()

    def __del__(self):
        self.mongoDB.disconnect()

    def get_source_path(self) -> str:
        """Get initialized path to Json Source File

        Returns:
            string: Path to Json Source File
        """
        return self.source_path

    def preprocess(self):
        """Preprocess (Read Json, Clean , Tokenize, Parse, insert DB) Json Source File
        """
        self.__read_file_content()

    def preprocess_sentence(self, sentence: str) -> Quintuple:
        """Preprocess input sentence
        """

        return self.__parse_depandance(sentence)

    def __read_file_content(self) -> None:
        """Read Lines of Json, Clean Content, Tokenize Sentences, Parse Quintuples and Insert into Database (id = content_id + sentence_no)
        """
        lines = []
        with open(self.get_source_path(), "r") as f:
            lines = f.readlines()[:10000]
        self.__preprocess_lines(lines)

    def __preprocess_lines(self, lines: List[str]) -> None:
        """ iterate through document, where each line is a court-case which has multiple sentences"""

        meta = []
        ids: list[int] = []
        sents: list[str] = []
        quintuples = []
        line_count = 0

        all_docs = len(lines)
        for index, line in enumerate(lines):
            print(f"Index: {index}/{all_docs}")
            json_ = json.loads(line)
            content, meta_data, id = self.__extract_content(json_)

            # clean the data and make sentences
            cleaned_c = self.__clean_content(content)
            sentences = self.__sent_tokenize_content(cleaned_c)

            #iterate through sentences
            for sentence_no, sentence in enumerate(sentences):

                quintuple = self.__parse_depandance(sentence)

                #write in database
                self.__write_to_database(sentence=sentence, quintuple=quintuple, id=f"{id}_{sentence_no}")
                line_count += 1

                quintuples.append(quintuple.serialize())
                sents.append(sentence)
                ids.append(int(id))
                meta.append(meta_data)

                assert len(ids) == len(meta) == len(sents) == len(quintuples)

        self.write_csv(quintuples, ids=ids, sentences=sents, meta=meta)

        # END

    def write_csv(self, quintuples: list[dict[str, list[str]]], ids: list[int], sentences: list[str], meta: list[dict[str, str]]):
        """write parsed data to a csv file

        Args:
            quintuples (Quintuple): [description]
            id (int): [description]
            content (str): [description]
            meta_data (dict[str,str]): [description]
        """
        # quintuple_entries = [quintuple.serialize() for quintuple in quintuples]
        # print(quintuple_entries)
        df_quintuples = pd.DataFrame(quintuples, columns=["subjects", "iobjects", "root", "dobjects", "rest"])
        df_meta = pd.DataFrame({"id": [f"ID{id}_LINE{i}" for i, id in enumerate(ids)],
                                "meta_data": meta,
                                "sentence": sentences,
                                "quintuples": quintuples})

        df = df_quintuples.join(df_meta, how="outer")
        df.to_csv(self.config["parsing_results_file"])

    def __extract_content(self, json: dict[str, str]):
        """reads content, id and metadata from json

        Args:
            json (dict[str,str]): [description]

        Returns:
            [type]: [description]
        """

        id = json["id"]

        # make sure content is in keys
        keys = json.keys()
        assert "content" in keys
        content = json.pop('content')
        # metadata is all the rest of th
        # e document
        meta_data: dict[str, str] = json
        return content, meta_data, id

    def __clean_content(self, content: str) -> str:
        """Clean Content -> Replace html Tags & umlauts

        Args:
            content (str): Text to be replaced

        Returns:
            str: Cleaned Text
        """
        return clean(content)

    def __sent_tokenize_content(self, cleaned_content: str) -> list[str]:
        """Tokenize Content -> Split Sentences & seperate them

        Args:
            cleaned_content (str): Already Cleaned Text to be tokenized

        Returns:
            list: List of (str) Sentences
        """
        return self.ST.tokenize(cleaned_content)

    def __parse_depandance(self, sentence: str) -> Quintuple:
        """Parse Depandances -> Parse sentence into it's Depandaces (Quintuples)

        Args:
            sentence (string): Sentence to be parsed

        Returns:
            string: Quintuples of the given Sentence
        """
        return self.QP.store_quintuple(sentence)

    def __write_to_database(self, sentence: str, quintuple: Quintuple, id: str) -> None:
        """Write Data into Databse

        Args:
            sentence (str): Cleaned Sentence
            quintuple (str): Parsed Quintuples of Cleaned Sentence
            id (str): unique Id -> [content_id] + _ + [Sentence_id]
        """
        self.mongoDB.create(
            {'id': id, 'sentence': sentence, 'quintuple': quintuple.serialize()})


if __name__ == "__main__":
    pp = Preprocessor()
    pp.preprocess()
