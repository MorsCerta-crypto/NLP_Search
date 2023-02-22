from os import environ

from Parser.parser import Quintuple
from pymongo import MongoClient


class MongoDBController():

    def __init__(self, url: str = "localhost", port: int = 27017, database: str = "NLPDB", collection: str = "CourtDecisions", results_limit: int = 20):
        if "MONGODB_HOST" in environ:
            self.url = environ['MONGODB_HOST']
        else:
            self.url = url
        # Port aus docker-compose.yml verwenden, falls vorhanden
        if "MONGODB_PORT" in environ:
            self.port = int(environ['MONGODB_PORT'])
        else:
            self.port = port
        self.database = database
        self.collection = collection
        self.results_limit = results_limit

    def connect(self):
        """Establishes a connection to a database.
        """
        self.client = MongoClient(self.url, self.port)
        self.db = self.client[self.database]
        self.active_collection = self.db[self.collection]

    def disconnect(self):
        """Closes the connection to the MongoDB.

        Raises:
            ConnectionError: Error is raised when the client isn't connected to any database.
        """
        try:
            self.client.close()
        except AttributeError:
            raise ConnectionError("Client is not connected to any database.")

    def create(self, entry_data: dict):
        """Creates a new entry in the active collection.

        Args:
            entry_data (dict): Data of the entry that is created.
        """
        self.active_collection.insert_one(entry_data)

    def update(self, search_param: str, search_value: str, updated_value: str):
        """Updates an entry in the active collection.

        Args:
            search_param (str): Parameter that is searched for in database.
            search_value (str): Value that has to match.
            updated_value (str): Value that the entry is updated by
        """
        self.active_collection.replace_one({
            search_param: {'$eq': search_value}
        }, updated_value)

    def remove(self, search_param: str, search_value: str):
        """Removes an entry from the active collection

        Args:
            search_param (str): Parameter that is searched for in database.
            search_value (str): Value that has to match.
        """
        self.active_collection.delete_one({
            search_param: {'$eq': search_value}
        })

    def remove_collection(self):
        """Removes the whole active collection from the database.
        """
        self.db.drop_collection(self.active_collection)

    def search_full_text(self, value: str) -> list:
        # search for a value in the content field
        query = {"sentence": {"$regex": value}}
        return self.__search(query)

    def search_field(self, field: str, value: str) -> list:
        query = {field: value}
        return self.__search(query)

    def search_parsed(self, quintuple: Quintuple) -> list:
        serialized_quintuple = quintuple.serialize()
        query_params = []
        for field in serialized_quintuple.keys():
            if len(serialized_quintuple[field]) != 0:
                query_param = {f"quintuple.{field}": {"$in": serialized_quintuple[field]}}
                query_params.append(query_param)
        query = {"$and": query_params}
        return self.__search(query)

    def get_sentences(self, num_sentences) -> list:
        query = {}
        sentences = []
        for entry in self.active_collection.find(query).limit(num_sentences):
            entry.pop('_id')
            sentences.append(entry["sentence"])
        return sentences

    def get_labels(self, num_labels) -> list:
        query = {}
        sentences = []
        for entry in self.active_collection.find(query).limit(num_labels):
            entry.pop('_id')
            sentences.append(entry["id"])
        return sentences

    def __search(self, query: dict) -> list:
        """Function used to search for with a query in a collection.

        Args:
            query (dict): Query that is used for the search.

        Returns:
            list: List of result entries.
        """
        results = []
        for entry in self.active_collection.find(query).limit(self.results_limit):
            entry.pop('_id')
            results.append(entry)
        return {"results": results}
