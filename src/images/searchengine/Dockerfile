# syntax=docker/dockerfile:1

FROM python:3.9.9-bullseye

WORKDIR /home/app/searchengine

COPY /searchengine /home/app/searchengine
COPY /frontend /home/app/frontend

# install all required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# install searchengine package
RUN python setup.py install
# install spacy language package
RUN python -m spacy download de_core_news_lg
RUN python -m nltk.downloader stopwords
ENV FLASK_APP Interface/interface.py

CMD [ "flask", "run", "--host", "0.0.0.0"]
