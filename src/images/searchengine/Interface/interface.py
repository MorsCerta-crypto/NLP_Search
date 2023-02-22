import os
import time

from flask import Flask, render_template, request
from RequestHandler.request_handler import RequestHandler
from Utilities.read_config import read_config

start_time = time.time()
config = read_config()
# initialize flask app and request handler
app = Flask(__name__, template_folder=os.path.abspath(config["templates_folder"]), static_folder=os.path.abspath(config["static_folder"]))
reqHdlr = RequestHandler(from_db=True)

quintuple_fields = ["root", "subjects", "dobjects", "iobjects", "rest"]
stop_time = time.time()
print("Elapsed time for Flask Webserver setup: ", stop_time - start_time)


@app.route('/', methods=['GET'])
def home():
    return render_template("index.html", QuinTuple=quintuple_fields)


@app.route('/full', methods=['POST', 'GET'])
def sentence_full_text_search() -> str:
    """Function used for a full text search.

    Returns:
        str: Result(s) of the search.
    """
    #text = request.args.get('text')
    if request.method == 'POST':
        text = request.form['text']

        return render_template("search_results.html", content=reqHdlr.full_text_search(text), SearchType=request.form['search-type'], QuinTuple=quintuple_fields, SearchInput=text)

    return render_template("search_results.html", content=[], QuinTuple=quintuple_fields)
    # return reqHdlr.full_text_search(text)


@app.route('/field', methods=['POST', 'GET'])
def quintuple_field_search() -> str:
    """Function used for a field search.

    Returns:
        str: Result(s) of the search.
    """
    if request.method == 'POST':
        field = request.form['field']
        value = request.form['text']
        if field in quintuple_fields:
            return render_template("search_results.html", content=reqHdlr.field_search(field, value),
                SearchType=request.form['search-type'], QuinTuple=quintuple_fields, SearchInput=value, SearchFieldIdx=field)
        else:
            return f"Wrong value for query parameters.\nField: {field}\nValue: {value}"

    return render_template("search_results.html", content=[], QuinTuple=quintuple_fields)


@app.route('/parsed', methods=['POST', 'GET'])
def parsed_search() -> str:
    """Function used for a parsed search.

    Returns:
        str: Result(s) of the search.
    """
    if request.method == 'POST':
        text = request.form['text']
        return render_template("search_results.html", content=reqHdlr.parsed_search(text), SearchType=request.form['search-type'], QuinTuple=quintuple_fields, SearchInput=text)

    return render_template("search_results.html", content=[], QuinTuple=quintuple_fields)


@app.route('/id', methods=['GET'])
def get_decision_by_id() -> str:
    decision_id = int(request.args.get('id'))
    return reqHdlr.id_search(decision_id)


@app.route('/suggestions', methods=['GET'])
def get_suggestion() -> str:
    text = request.args.get('text')
    return reqHdlr.get_suggestions(text)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
