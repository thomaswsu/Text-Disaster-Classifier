from sklearn.model_selection import train_test_split, KFold
from classifier import vectorize_data1, vectorize_data2, vectorize_data3, vectorize_data4, \
   SVM, NB, KNN, RF, iterative_train_RF, SGCD, iterative_train_SGCD
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import FeatureUnion, Pipeline
import numpy 
import pickle
import Tweet
from sklearn.svm import SVC

def get_tweets_urls_and_hashtags(tweets: list) -> list:
    """ Takes in a list of Tweets. Returns a list of tweets, urls. and hastags for each item in tweets list.
    Looks like: [[list of tweets] [list of urls] [list of hashtags]]
    Length of returned list is equal to the length of input list.
    """
    """ DATA PARSING HAS BEEN REMOVED TO PROTECT DATA
    """
    return()

def tokenization_train(training_data, folds: int, portionOfDataSet: float):
    """ This test implements tokinization and cross-validation.
    Currently using the "Bag of Words" approch to vectorize the data. 
    """
    X = [] # training data
    y = [] # Class labels. Disaster or not disaster 
    print("Using {} % of data set...".format(portionOfDataSet * 100))
    print("Parsing text...")
    for i in range(int(len(training_data.index) * portionOfDataSet)):
        X.append(str(training_data["ttext"][i]))
        y.append(int(training_data["Donation"][i]))
    X = get_tweets_urls_and_hashtags(X)
    

    print("Vectorizing data set...")
    vectorized_X = []
    vectorized_X.append(vectorize_data3(X[0]))
    vectorized_X.append(vectorize_data3(X[1]))
    vectorized_X.append(vectorize_data3(X[2]))
    y = numpy.array(y)

    print("Splitting up data set...")
    kf = KFold(n_splits = folds)
    kf.get_n_splits(vectorized_X[0])
    
    i = 1
    results = []
    for train_indexes, test_indexes in kf.split(vectorized_X[0]): 
        """ train_indexes contains the list of numpy array of indexes for training and testing data. 
        The first element of train_indexes is the training indexes and the second is the testing. 
        """
        X_train, X_test = [vectorized_X[0][train_indexes], vectorized_X[1][train_indexes],\
             vectorized_X[2][train_indexes]], \
                 [vectorized_X[0][test_indexes], vectorized_X[1][test_indexes], vectorized_X[2][test_indexes]]
        y_train, y_test = y[train_indexes], y[test_indexes]

        print("RF Iteration {} of {}...".format(i, folds))
        clf = RF(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print("RF Results:")
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))  
        print(accuracy_score(y_test, y_pred))
        results.append(accuracy_score(y_test, y_pred))
        i += 1
    print(results)
    
    
    


def test1(training_data, portionOfDataSet: float, test_ratio: float):
    """ Test 1 runs SVM, NB, KNN, RF without cross validation. 
    Uses the "Bag of Words" method to vectorize the data.
    """
    X = [] # training data
    y = [] # Class labels. Disaster or not disaster 
    print("Using {} % of data set...".format(portionOfDataSet * 100))
    print("Parsing text...")
    for i in range(int(len(training_data.index) * portionOfDataSet)):
        X.append(str(training_data["ttext"][i]))
        y.append(int(training_data["Disaster"][i]))

    print("Vectorizing Data...")
    vectorized_data = vectorize_data1(X)

    print("Spliting data into training and testing data...")
    X_train, X_test, y_train, y_test = train_test_split(vectorized_data, y, \
        test_size = test_ratio, random_state = 0) 

    print("Training SVM Model...")
    clf_svm = SVM(X_train, y_train) 

    y_pred_svm = clf_svm.predict(X_test) 
    print("SVM Results:")
    print(confusion_matrix(y_test,y_pred_svm))  
    print(classification_report(y_test,y_pred_svm))  
    print(accuracy_score(y_test, y_pred_svm)) 
    

    print("Training GaussianNB Model...")
    clf_NB =  NB(X_train, y_train)

    y_pred_NB = clf_NB.predict(X_test) 
    print("GaussianNB Results:")
    print(confusion_matrix(y_test,y_pred_NB))  
    print(classification_report(y_test,y_pred_NB))  
    print(accuracy_score(y_test, y_pred_NB))

    print("Training KNN Model...")
    clf_KNN = KNN(X_train, y_train)

    y_pred_KNN = clf_KNN.predict(X_test)
    print("KNN Results:")
    print(confusion_matrix(y_test,y_pred_KNN))  
    print(classification_report(y_test,y_pred_KNN))  
    print(accuracy_score(y_test, y_pred_KNN))
    
    print("Training RF Model...")
    clf_RF = RF(X_train, y_train)

    y_pred_RF = clf_RF.predict(X_test)
    print("RF Results:")
    print(confusion_matrix(y_test,y_pred_RF))  
    print(classification_report(y_test,y_pred_RF))  
    print(accuracy_score(y_test, y_pred_RF))

def test2(training_data, folds: int, portionOfDataSet: float):
    """ Implemented with cross validation
    """
    X = [] # training data
    y = [] # Class labels. Disaster or not disaster 
    print("Using {} % of data set...".format(portionOfDataSet * 100))
    print("Parsing text...")
    for i in range(int(len(training_data.index) * portionOfDataSet)):
        X.append(str(training_data["ttext"][i]))
        y.append(int(training_data["Donation"][i]))

    print("Vectorizing Data...")
    vectorized_data = vectorize_data1(X)
    y = numpy.array(y)
    """ Split the data into training and testing data. 
    Notre the true_X_test and true_y_test are never going to be used for training. 
    """

    kf = KFold(n_splits = folds)
    kf.get_n_splits(vectorized_data)
    
    i = 1
    results = []
    for train_indexes, test_indexes in kf.split(vectorized_data): 
        """ train_indexes contains the list of numpy array of indexes for training and testing data. 
        The first element of train_indexes is the training indexes and the second is the testing. 
        """
        X_train, X_test = vectorized_data[train_indexes], vectorized_data[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

        print("SVM Iteration {} of {}...".format(i, folds))
        clf = SVM(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print("SVM Results:")
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))  
        print(accuracy_score(y_test, y_pred))
        results.append(accuracy_score(y_test, y_pred))
        i += 1
    print(results)

    for train_indexes, test_indexes in kf.split(vectorized_data): 
        """ train_indexes contains the list of numpy array of indexes for training and testing data. 
        The first element of train_indexes is the training indexes and the second is the testing. 
        """
        X_train, X_test = vectorized_data[train_indexes], vectorized_data[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

        print("RF Iteration {} of {}...".format(i, folds))
        clf = RF(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print("RF Results:")
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))  
        print(accuracy_score(y_test, y_pred))
        results.append(accuracy_score(y_test, y_pred))

    for train_indexes, test_indexes in kf.split(vectorized_data): 
        """ train_indexes contains the list of numpy array of indexes for training and testing data. 
        The first element of train_indexes is the training indexes and the second is the testing. 
        """
        X_train, X_test = vectorized_data[train_indexes], vectorized_data[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

        print("KNN Iteration {} of {}...".format(i, folds))
        clf = KNN(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print("KNN Results:")
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))  
        print(accuracy_score(y_test, y_pred))
        results.append(accuracy_score(y_test, y_pred))

    for train_indexes, test_indexes in kf.split(vectorized_data): 
        """ train_indexes contains the list of numpy array of indexes for training and testing data. 
        The first element of train_indexes is the training indexes and the second is the testing. 
        """
        X_train, X_test = vectorized_data[train_indexes], vectorized_data[test_indexes]
        y_train, y_test = y[train_indexes], y[test_indexes]

        print("NB Iteration {} of {}...".format(i, folds))
        clf = NB(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        print("NB Results:")
        print(confusion_matrix(y_test,y_pred))  
        print(classification_report(y_test,y_pred))  
        print(accuracy_score(y_test, y_pred))
        results.append(accuracy_score(y_test, y_pred))

    """
    filename = 'RF_Cross_Validation.sav' # Save the models 
    pickle.dump(clf_RF, open(filename, 'wb'))

    filename = 'SGCD_Cross_Validation.sav'
    pickle.dump(clf_SGCD, open(filename, 'wb'))
    """

def test4(training_data, folds: int, portionOfDataSet: float, test_ratio: float):
    """ This is with more tokens 
    """
    X = [] # training data
    y = [] # Class labels. Disaster or not disaster 
    Tweets = []
    print("Using {} % of data set...".format(portionOfDataSet * 100))
    print("Parsing text...")
    for i in range(int(len(training_data.index) * portionOfDataSet)):
        Tweets.append(Tweet.Tweet(training_data["statusid"], training_data["ttext"]))        


""" Sources:
https://machinelearningmastery.com/k-fold-cross-validation/
"""
