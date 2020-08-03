from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model

""" Don't forget to install sklearn
"""

def vectorize_data1_tokenization(data: list) -> list:
    """ We haven't really defined how to make this yet...
    """
    return()

def vectorize_data1(data: list) -> list:
    """ Convert a list of string into vectors so it can be used training
    Uses the "bag of words" method. 
    """
    vectorizer = CountVectorizer(lowercase = False, stop_words = "english")  # filter out common English words 
    # Here we convert strings to vectors using the bag of words method. Not super sure on the method.
    vectorized_samples = vectorizer.fit_transform(data).toarray()
    return(vectorized_samples)

def vectorize_data2(data: list) -> list:
    """ This vectorizes by word count
    """
    vectorizer = TfidfVectorizer(lowercase = False, stop_words = "english")
    vectorized_samples = vectorizer.fit_transform(data).toarray()
    return(vectorized_samples)

def vectorize_data3(data: list) -> list:
    """ Vectorize by some kind of hash (that I don't really know)
    DOES NOT CURRENT WORK ON LAPTOP because memory usage is crazy 
    Figure out how to reduce?
    """
    vectorizer = HashingVectorizer(lowercase = False ,stop_words = "english")
    vectorized_samples = vectorizer.fit_transform(data).toarray()
    return(vectorized_samples)

def vectorize_data4(data: list) -> list:
    vectorizer = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False)
    vectorized_samples = vectorizer.transform(data).toarray()

    return(vectorized_samples)

def vectorize_data5(data: list) -> list:
    """ Super experiemental way to vectorize data
    """
    return()

def SVM(X: list, y: list):
    """ Support Vector Machine \n
    Input is array of samples (A list of Tweets) and a list of tags. Ex: [0, 1, 1, 0, 1]
    """

    """ Issue is that we need to hash the input text into a float somehow. What are some good methods? 
    Will try "bag of words" for now.
    """

    clf = svm.SVC(gamma = "scale") # could probs reseach more about the "gamma" flag 
    clf.fit(X, y) 
    return(clf)


def NB(X: list, y: list):
    clf = GaussianNB()
    clf.fit(X, y)
    return(clf)

def KNN(X: list, y: list):
    clf = KNeighborsClassifier()
    clf.fit(X, y)
    return(clf)

""" Functions below can work with iterative training
"""

def RF(X: list, y: list):
    clf = RandomForestClassifier(n_estimators = 100, warm_start=True) # add verbose = 3 flag to see process
    clf.fit(X, y)
    return(clf)

def iterative_train_RF(RF_model, X: list, y: list, increase_estimators: int):
    RF_model.n_estimators += increase_estimators
    RF_model.fit(X, y)
    return(RF_model)

def SGCD(X: list, y: list):
    """ Apparently its like SVM
    """
    clf_SGCD = linear_model.SGDClassifier()
    clf_SGCD.fit(X, y)
    return(clf_SGCD)

def iterative_train_SGCD(SGCD_model, x: list, y: list):
    SGCD_model.partial_fit(x, y)
    return(SGCD_model)

if __name__ == "__main__": # debugging purposes 
    pass

""" Sources!
https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
https://chrisalbon.com/machine_learning/basics/perceptron_in_scikit-learn/
"""
