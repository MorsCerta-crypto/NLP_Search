
from dataclasses import asdict, dataclass
from typing import List, Union
from spacy import explain, load, displacy
from spacy.tokens.span import Span


@dataclass
class Quintuple:
    root: list[str]
    subjects: list[str]
    dobjects: list[str]
    iobjects: list[str]
    rest: list[str]

    @staticmethod
    def get_dependencies() -> dict[str, List[str]]:
        return dict({
            "root": ["ROOT"],
            "subjects": ["sb", "sp", "sbp"],    # Subject
            "dobjects": ["oa"],                 # direct Objects
            "iobjects": ["op", "og", "oc"],     # indirect Objects
            # "conjunctions": ["ju", "cj", "cm", "cc", "cd"],
            "rest": []
        })

    @property
    def get_root(self) -> Union[List[str], None]:
        return self.get_dependencies().get("ROOT")

    @property
    def get_subjects_types(self) -> Union[List[str], None]:
        return self.get_dependencies().get("sbjs")

    @property
    def get_dobjects_types(self) -> Union[List[str], None]:
        return self.get_dependencies().get("dobjects")

    @property
    def get_iobjects_types(self) -> Union[List[str], None]:
        return self.get_dependencies().get("iobjects")

    @property
    def get_conjunctions_types(self) -> Union[List[str], None]:
        return self.get_dependencies().get("conjunctions")

    def serialize(self):
        """returns dict with values from given Quintuple class"""
        return asdict(self)


class QuintupleReducer:
    """Parses split sentences into quintuples"""
    quintuple: Quintuple
    default_dict: dict[str, List[str]]
    conjunctions: List[str] = ["ju", "cj", "cm", "cc", "cd"]

    def __init__(self, bad_keys: list[str] = ["punkt"]) -> None:
        self.bad_keys = ["punct"]
        # load defaultdict with map of quintuple keys
        self.dependency_map = Quintuple.get_dependencies()
        q = Quintuple([], [], [], [], [])
        self.default_dict = q.serialize()
        self.nlp = load("de_core_news_lg")

    def has_conjunction(self, dependency_dict: dict[str, List[str]]) -> bool:
        ans = any([key for key in dependency_dict.keys() if key in self.conjunctions])
        return ans

    def _get_dict(self, sent: Span) -> dict[str, List[str]]:
        """store sentence dependencies and text of each token in a dict"""
        dependency_dict: dict[str, list[str]] = {}
        # print(f"{sent=}")
        for child in sent:  # type: ignore
            current_dependency: str = child.dep_
            text: str = child.text

            if current_dependency in dependency_dict.keys():
                # append dependency as key and test as value to dict
                dependency_dict[current_dependency].append(text)
            else:
                # extend dict
                dependency_dict[current_dependency] = [text]
        return dependency_dict

    def store_quintuple(self, sentence: str) -> Quintuple:
        """ inserts dep, lemma - touple into a dictionary depending on dependencies"""

        # find all dependencies of sent
        sent = self.nlp(sentence)
        deps = self._get_dict(sent)

        # sentence has conjuntion structure
        has_conjunction = self.has_conjunction(deps)

        # reduce dependencies to certain set
        quintuple = self._reduce_keys(deps)
        return quintuple

    def _reduce_keys(self, deps: dict[str, List[str]]) -> Quintuple:
        """ reduce multiple subj, objs and preds to single name"""
        reduced_dict: dict[str, list[str]] = {
            x: list() for x in self.default_dict.keys()}

        for dependency, text in deps.items():

            new_key = self._get_key_from_value(dependency)

            # exclude punkt and others #TODO
            if dependency in self.bad_keys:
                continue

            # make sure text is not empty
            if text == [""]:
                continue
            reduced_dict[new_key] = reduced_dict[new_key] + text

        quintuple = Quintuple(**reduced_dict)

        return quintuple

    def _get_key_from_value(self, value: str) -> str:
        """find key in input dict that contains value"""

        for k, v in self.dependency_map.items():
            if value in v:
                return k
        else:
            if value:
                return "rest"
            else:
                raise ValueError("empty value")

    def explain_dependecies(self) -> None:
        deps = ["dobj", "iobj", "ROOT", "ac", "adc", "ag", "ams", "app", "avc", "cc", "cd", "cj", "cm", "cp", "cvc", "da", "dep", "dm", "ep", "ju", "mnr", "mo",
                "ng", "nk", "nmc", "oa", "oc", "og", "op", "par", "pd", "pg", "ph", "pm", "pnc", "punct", "rc", "re", "rs", "sb", "sbp", "svp", "uc", "vo"]
        for elem in deps:
            print("\n ", elem, "means: ", explain(elem))


class SpacyVisualizer:
    """class for visiualizing the spacy dependencies of a sentence"""

    def __init__(self):
        self.nlp = load("de_core_news_lg")

    def get_html_of_spacy_dependences(self, sentence: str):
        """ take a sentence as input and return the html-visualisation of the dependencies"""
        doc = self.nlp(sentence)
        return displacy.render(doc, style="dep")


def main():
    """
    this function shows how to use the dependency-parser
    class QuintupleReducer and SpacyVisualizer
    """

    # Usage Example:
    sentence = "This is a sentence"
    QuintupleRed = QuintupleReducer()
    # QP.explain_dependecies()
    quintuples: dict[str, str] = QuintupleRed.store_quintuple(sentence).serialize()

    # Visualisation of dependencies
    sv = SpacyVisualizer()
    html = sv.get_html_of_spacy_dependences(sentence)

    # set to true for writing output to file
    if False:
        with open("html.text", "w+") as f:
            f.write(html)
        # open html file with browser


if __name__ == "__main__":
    main()
