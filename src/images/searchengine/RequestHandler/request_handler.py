# import pandas as pd
import os
from re import I

from MongoDBController.mongodb_controller import MongoDBController
from n_gram.inference import Inference
from n_gram.train_input_prediction import run_training
from preprocessing.preprocessor import Preprocessor
from Parser.parser import SpacyVisualizer
from Utilities.read_config import read_config


class RequestHandler():
    def __init__(self, from_db: bool = True) -> None:
        self.from_db = from_db
        self.config = read_config()
        self.parsing_results = self.config["parsing_results_file"]
        self.mongodb = MongoDBController()
        self.preprocessor = Preprocessor()
        self.spacy_vis = SpacyVisualizer()

        if not os.path.exists("model.bin"):
            run_training(learn_up_to=8, from_db=self.from_db)

        self.n_gram_inference = Inference()

    def full_text_search(self, full_text: str) -> str:
        self.mongodb.collection = "CourtDecisions"
        self.mongodb.connect()  # connect to MongoDB
        results = self.mongodb.search_full_text(full_text)
        self.mongodb.disconnect()
        results = self.__create_visualization(results)
        return results

    def field_search(self, field: str, value: str) -> str:
        self.mongodb.collection = "CourtDecisions"
        self.mongodb.connect()  # connect to MongoDB
        results = self.mongodb.search_field(f"quintuple.{field}", value)
        self.mongodb.disconnect()
        results = self.__create_visualization(results)
        return results

    def parsed_search(self, text: str) -> str:
        self.mongodb.collection = "CourtDecisions"
        self.mongodb.connect()  # connect to MongoDB
        # preprocess text to quintuple
        quintuple = self.preprocessor.preprocess_sentence(text)
        results = self.mongodb.search_parsed(quintuple)
        self.mongodb.disconnect()
        results = self.__filter_results(results)
        results = self.__create_visualization(results)
        return results

    def id_search(self, decision_id: int):
        self.mongodb.collection = "RawData"
        self.mongodb.connect()  # connect to MongoDB
        results = self.mongodb.search_field("id", decision_id)
        self.mongodb.disconnect()
        return results

    def get_suggestions(self, text: str):
        return {"results": self.n_gram_inference.predict_on_context_n_grams(text.split())}

    def __filter_results(self, results: dict) -> dict:
        """Function used to filter results for unique IDs to avoid duplicates

        Args:
            results (dict): Results from search

        Returns:
            dict: Filtered results with unique IDs.
        """
        results_list = results["results"]
        unique_ids = list(set([id["id"] for id in results_list]))
        filtered_results = []
        for i in unique_ids:
            tmp_results = []
            for d in results_list:
                if d["id"] == i:
                    tmp_results.append(d)
            filtered_results.append(tmp_results[0])
        return {"results": filtered_results}

    def __create_visualization(self, results: dict) -> dict:
        for result in results["results"]:
            result["graph"] = self.spacy_vis.get_html_of_spacy_dependences(result["sentence"])
        return results