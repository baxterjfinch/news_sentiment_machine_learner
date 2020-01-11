import csv
import json
import requests
import pydotplus
import numpy as np
import pandas as pd
from IPython.display import Image
from sklearn import preprocessing
from pandas import read_csv, set_option
from sklearn.externals.six import StringIO
from sklearn.feature_selection import chi2
from numpy import loadtxt, set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


API_KEY = ""

def get_articles(section=None, query=None):
    response = requests.get("https://newsapi.org/v2/everything?q=bitcoin&pageSize=100&apiKey={}".format(API_KEY))
    content = json.loads(response.content)

    total_results = content["totalResults"]
    print("Total results: {}".format(total_results))
    articles = content["articles"]
    print(len(articles))
    # print("Total Results: {}".format(total_results))

    return articles

def convert_to_csv(articles):
    sources = organize_sources(articles)
    # print(sources)
    # print("Sources {}".format(sources))

    with open('article_csv.csv', mode='w') as article_csv:
        fieldnames = ['positive_sentiment', 'negative_sentiment', 'opinionated', 'bias']
        writer = csv.DictWriter(article_csv, fieldnames=fieldnames)

        for key, value in sources.items():
            for article in value:
                #'title': '{}'.format(article["title"]),
                writer.writerow({'positive_sentiment': article["positive_sentiment"], 'negative_sentiment': article["negative_sentiment"], 'opinionated': article["opinionated"], 'bias': article["bias"]})


def organize_sources(articles):
    sources = []
    organize_sources = {}

    for article in articles:
        if article["source"]["name"] not in sources:
            sources.append(article["source"]["name"])
            organize_sources[article["source"]["name"]] = []

        print(article["source"]["name"])
        words = ''
        if (article["description"]):
            words += article["description"]

        if (article["title"]):
            words += article["title"]

        if (article["content"]):
            words += article["content"]

        pos, neg, bias, opinionated = get_sentiment(words)

        organize_sources[article["source"]["name"]].append({
            "title": article["title"],
            "positive_sentiment": pos,
            "negative_sentiment": neg,
            "bias": bias,
            "opinionated": opinionated
        })

    print(organize_sources)
    return organize_sources


def get_sentiment(upper_words):
    # print("\nNew Section\n")

    pos = 0
    neg = 0
    opinionated = 0

    upper_words = upper_words.split()
    words = []

    for word in upper_words:
        words.append(word.lower())

    with open('positive-words.txt') as pf:
        for pos_line_word in pf.readlines():
            # print(pos_line_word)
            # print("\n", words)

            if pos_line_word.strip() in words:
                # print("POSITIVE SENTIMENT: " + pos_line_word)
            # if word == line_word:
            #     print("POSITIVE SENTIMENT" + word)
                pos += 1
                opinionated += 1

    with open('negative-words.txt') as nf:
        for neg_line_word in nf.readlines():
            if neg_line_word.strip() in words:
                # print("NEGATIVE SENTIMENT: " + neg_line_word)
                neg -= 1

                opinionated += 1

    # print("postive: {} negative: {} opinionated: {}".format(pos, neg, opinionated))
    # print("\n-----------")

    bias = 0
    if pos > abs(neg):
        bias = 1

    print(bias)
    return pos, neg, bias, opinionated


articles = get_articles()
csv = convert_to_csv(articles)

# set_option('max_colwidth', 1000)
names = ['positive_sentiment', 'negative_sentiment', 'opinionated', 'bias']
dataframe = pd.read_csv("article_csv.csv", names=names)
print(dataframe.head())
print(dataframe.columns)

feature_cols = ['positive_sentiment', 'negative_sentiment', 'opinionated']
X = dataframe[feature_cols] # Features
y = dataframe["bias"] # Target variables

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

clf = DecisionTreeClassifier(criterion="entropy", max_depth=15)
clf = clf.fit(X_train,y_train)

###################
# Make a prediction
y_pred = clf.predict(X_test)


# Get accuracy score, confusion matrix, and classification report
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
   special_characters=True,feature_names = feature_cols)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('bitcoin_news.png')
Image(graph.create_png())

# print(dataframe)
