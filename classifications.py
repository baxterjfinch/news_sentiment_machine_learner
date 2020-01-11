import sklearn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

data = load_breast_cancer()

def organize_data(data):
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']

    return label_names, labels, feature_names, features

def evaluate_data_gaussian(train, train_labels):
    gnb = GaussianNB()
    gnb.fit(train, train_labels)
    return gnb

def prediction_tests(gnb, test):
    return gnb.predict(test)

label_names, labels, feature_names, features = organize_data(data)

train, test, train_labels, test_labels = train_test_split(features,labels,test_size = 0.40, random_state = 42)

model = evaluate_data_gaussian(train, train_labels)
preds = prediction_tests(model, test)
print(preds)

### Get accuracy
print(accuracy_score(test_labels, preds))
