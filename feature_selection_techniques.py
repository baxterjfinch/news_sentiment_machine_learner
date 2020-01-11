from sklearn.feature_selection import SelectKBest, RFE
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA

from numpy import loadtxt, set_printoptions

def univariate_selection(X, Y, feature_size, precision):
    test = SelectKBest(score_func=chi2, k=4)
    fit = test.fit(X,Y)
    set_printoptions(precision=2)
    # print(fit.scores_)
    return fit.transform(X)

def recursive_feature_elimination(array, X, Y, amount):
    model = LogisticRegression()
    rfe = RFE(model, amount)
    return rfe.fit(X, Y)

def principal_component_analysis(array, X, Y, amount):
    print("Keeping {}".format(amount))
    pca = PCA(n_components = amount)
    fit = pca.fit(X)
    print("Explained Variance: {}".format(fit.explained_variance_ratio_))
    print(fit.components_)
    return fit


def feature_importance(array, X, Y):
    model = ExtraTreesClassifier()
    model.fit(X, Y)
    print(model.feature_importances_)
    return model
