import quandl
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.externals.six import StringIO
from IPython.display import Image
import math
import pydotplus
from sklearn import model_selection
import pickle
import time
import datetime as dt

APIKEY="7SSnVKnZMR59ZxqGv5PZ"
TS = time.time()

def authenticate():
    quandl.ApiConfig.api_key = APIKEY

def get_bitcoin_data(api):
    for i in range(500):
        print(i)
        from_date, to_date = get_previous_day()

        df = quandl.get("BCHARTS/BITSTAMPEUR", start_date=from_date, end_date=to_date)

        df["Change %"] = [percentage_change(row) for ind, row in df.iterrows()]

        with open('paged_data.csv', 'a') as f:
            df.to_csv(f, header=None, index=False)

def get_previous_day():
    global TS
    to_date = dt.datetime.utcfromtimestamp(TS).strftime("%Y/%m/%d")
    TS = TS - 86400
    frmt_date = dt.datetime.utcfromtimestamp(TS).strftime("%Y/%m/%d")

    return frmt_date, to_date

def save_paged_data(data):
    with open('paged_data.csv','a') as fd:
        fd.write(str(data))

def percentage_change(x):

    open = x["Open"]
    closed = x["Close"]
    percentage = 0

    if open > closed:
        percentage = ((open - closed) / open) * -1
    elif closed > open:
        percentage = (closed - open) / closed
    else:
        percentage = 0

    val = round(percentage, 2)

    if math.isnan(val):
        return 0.00
    else:
        return val

def save_model(clf):
    pickle.dump(clf, open('model.sav', 'wb'))

def load_model():
    return pickle.load(open('model.sav', 'rb'))

def build_decision_tree(df):
    df = pd.read_csv("paged_data.csv")
    feature_cols = ['Open', 'High', 'Low', 'Volume (BTC)', 'Volume (Currency)', 'Weighted Price']
    X = df[feature_cols]
    y = df['Change %']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1)

    clf = DecisionTreeRegressor(max_depth=10)
    clf = clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    #
    # result = confusion_matrix(y_test, y_pred)
    # print("Confusion Matrix:")
    # print(result)
    # result1 = classification_report(y_test, y_pred)
    # print("Classification Report:",)
    # print (result1)
    # result2 = accuracy_score(y_test,y_pred)
    # print("Accuracy:",result2)

    dot_data = StringIO()
    export_graphviz(clf, out_file=dot_data, filled=True, rounded=True,
       special_characters=True,feature_names = feature_cols)

    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('btc_decision_tree.png')

    Image(graph.create_png())
    save_model(clf)

def make_prediction(model):
    data = [[7622, 7700, 7455, 89, 674492, 7572]]
    y_pred = model.predict(data)
    print("PREDICTION\n", y_pred, data)

api = authenticate()
# get_bulk_data(api)
df = get_bitcoin_data(api)
build_decision_tree(df)
#
# model = load_model()
# make_prediction(model)
# save_to_csv(df)
