import re
import csv
import json
import requests
import pydotplus
import numpy as np
import pandas as pd
from pandas import DataFrame
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from nltk import sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import mysql.connector as mysql
from newsapi import NewsApiClient
from sklearn import preprocessing
from IPython.display import Image
from pandas import read_csv, set_option
from sklearn.externals.six import StringIO
from sklearn.feature_selection import chi2
from numpy import loadtxt, set_printoptions
from nltk.tokenize import TreebankWordTokenizer, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NYT API INFO
API_KEY = "kAPdkb5glWAXB1o9qlP2Odx7vO3IAv0S"
SECRET = "mYv2LWSR8MOPGY9t"

db = mysql.connect(
    host = "localhost",
    user = "root",
    passwd = "reojfist1",
    database = "nytsentiment"
)

class NYTApi():
    def __init__(self, api_key, year=None, month=None):
        self.api_key = api_key
        self.newsapi = None
        self.year = year
        self.month = month
        self.stop_words = set(stopwords.words('english'))

        # Array of docs arrays
        # [[archived_docs], [trending_docs]]
        self.retrieved_docs = []
        # dev purposes
        self.retrieved_doc = []

    def get_archive(self):
        response = requests.get('https://api.nytimes.com/svc/archive/v1/2019/{}.json?api-key={}'.format(self.month, self.api_key))
        # loaded = json.loads(archived_month)
        if response.status_code == 200:
            json_data = json.loads(response.content)
            meta = json_data["response"]["meta"]
            docs = json_data["response"]["docs"]
            # docs[i].keys() == ([
            # 'web_url', 'snippet', 'lead_paragraph',
            # 'print_page', 'blog', 'source', 'multimedia', 'headline',
            # 'keywords', 'pub_date', 'document_type', 'news_desk',
            # 'section_name', 'byline', 'type_of_material', '_id', 'word_count', 'score', 'uri'])

            # Take the whole response and appends to the class list
            # for dev purposes, I will just be taking and saving 1

            for i in range(5):
                response = requests.get(docs[i]['web_url'])
                bs = BeautifulSoup(response.content)
                text = bs.find("section", {'name':'articleBody'}).text
                # print(text)
                # self.retrieved_docs.append(docs)
                tokenized_words = self.tokenize_words_remove_stops(text)
                self.retrieved_docs.append({"id": i, "url": docs[i]['web_url'], "corpus": tokenized_words, "commons": {}})
                # print(self.retrieved_doc)
            # for doc in self.retrieved_docs:
                # print(doc["corpus"])
        else:
            print(response.status_code)

    def build_dataframe(self):
        d = list()


        for art in self.retrieved_docs:

            df = pd.DataFrame({"id": art["id"]}, index=[0])
            df['words'] = df.apply(lambda row: art["corpus"], axis=1)
            df['word_count'] = df['words'].apply(lambda x: len(x))
            df.word_count.describe()
            d.append(df)

        doc = pd.concat(d)
        print(doc["word_count"].describe())
        # freq = pd.Series(doc['words'])
        # for w in freq:
        #     print(w.value_counts())
        # print(freq)


    def tokenize_words_remove_stops(self, text):
        word_tokens = word_tokenize(text)
        tokenized_words = []
        tokenized_words.append([w for w in word_tokens if not w in self.stop_words and len(w) > 2])

        # stemmed = self.stem_words(tokenized_words[0])
        # return stemmed
        return tokenized_words[0]

    def stem_words(self, words):
        stemmed = []
        stemmer = SnowballStemmer("english")
        for word in words:
            stemmed.append(stemmer.stem(word))

        return stemmed

    def count_vectorize(self):
        vectorizer = CountVectorizer()
        corpus = []
        # print(self.retrieved_doc[0]["corpus"])
        for doc in self.retrieved_docs:
            corpus.extend(doc["corpus"])

        X = vectorizer.fit_transform(corpus)
        vector = vectorizer.transform(corpus)
        # x = vectorizer.vocabulary_
        # sorted_x = sorted(x.items(), key=lambda kv: kv[1])
        print('Full vector: ')
        print(vector.toarray())


    def vectorize_text(self):
        vect = TfidfVectorizer(self.retrieved_doc[0]["corpus"])

        tokenizer = TreebankWordTokenizer()
        vect.set_params(tokenizer=tokenizer.tokenize)

        #remove english stop words
        vect.set_params(stop_words='english')

        # include 1-grams and 2-grams
        vect.set_params(ngram_range=(1, 2))

        # ignore terms that appear in more than 50% of the documents
        vect.set_params(max_df=0.5)

        # only keep terms that appear in at least 2 documents
        vect.set_params(min_df=2)

        print(vect)





nytapi = NYTApi(API_KEY, year="2019", month="1")
nytapi.get_archive()
nytapi.build_dataframe()
# nytapi.count_vectorize()
