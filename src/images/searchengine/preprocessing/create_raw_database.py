import json
import time

from MongoDBController.mongodb_controller import MongoDBController
from Utilities.read_config import read_config

config = read_config()
DATABASE_FILE = config["database_file"]
DATABASE_PATTERN = ["id", "content"]


def create_raw_database(mongo_client: MongoDBController):
    print("Creating and filling database of court decisions.")
    with open(DATABASE_FILE, "r") as f:
        i = 0
        for line in f.readlines()[:10000]:
            i += 1
            print(i)
            data = json.loads(line)
            mongo_client.create({"id": data["id"], "content": data["content"]})
    print("Finished filling database.")


def main():
    m = MongoDBController(collection="RawData")
    m.connect()
    start_time = time.time()
    create_raw_database(m)
    end_time = time.time()
    m.disconnect()
    print("Elapsed time:", end_time - start_time)  # Output: Elapsed time: 214.7744104862213 seconds


if __name__ == "__main__":
    main()
