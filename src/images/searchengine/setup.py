import setuptools
import pathlib
from pkg_resources import parse_requirements
from sys import platform
from Utilities.read_config import read_config

CONFIG_FILE_MAC = "../searchengine/config.json"
if platform == "darwin":
    config = read_config(config_file=CONFIG_FILE_MAC)
else:
    config = read_config()

searchengine_root = str(pathlib.Path(__file__).parent.resolve())
readme_file = config["readme_file"]
requirements_file = config["requirements_file"]

with open(requirements_file, "r", encoding="utf-8") as file:
    requirements = [str(requirement) for requirement in parse_requirements(file)]

with open(readme_file, "r", encoding="utf-8") as file:
    long_description = file.read()

setuptools.setup(
    name="Searchengine",
    version=config["version"],
    author="",
    author_email="",
    description="Semantic Searchengine Package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    project_urls={
        "Bug Tracker": "",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": searchengine_root},
    packages=setuptools.find_packages(where=searchengine_root),
    install_requires=requirements,
    python_requires=">=3.6",
)
