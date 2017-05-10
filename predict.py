import os
import settings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

def read():
    train = pd.read_csv(os.path.join(settings.PROCESSED_DIR, "train.csv"))
    with open(os.path.join(settings.PROCESSED_DIR, "new_settings.txt"), "r") as f:
        new_dummy_features = f.read().splitlines() 
    return train, new_dummy_features

def get_y(train):
    if settings.PREDICTOR == ["CANCELLED"]:
        y = train[settings.PREDICTOR].as_matrix()
    else:
        y = train[['REASON_'+settings.reason]].as_matrix()
    c,r = y.shape
    y = y.reshape(c,)
    return y

def get_X(train,y):
    if settings.PREDICTOR == ["CANCELLED"]:
        X = train.drop(settings.PREDICTOR,axis=1)
    else:
        X = train.drop(['REASON_'+settings.reason],axis=1)
    return X

def split(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = settings.test_size,
                                                        random_state=1)
    return X_train,X_test,y_train,y_test

def train_model(X_train,X_test,y_train,y_test,new_dummy_features):
    X_train_dummy = X_train[new_dummy_features].as_matrix()
    X_test_dummy = X_test[new_dummy_features].as_matrix()
    X_train = X_train.drop(new_dummy_features,axis=1).as_matrix()
    X_test = X_test.drop(new_dummy_features,axis=1).as_matrix()
    #standarize non dummy features using training set
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    #recombine training sets
    X_train_final = np.concatenate((X_train_transformed,X_train_dummy),axis=1)
    # train a logistic regression model
    clf = LogisticRegression(random_state=1, class_weight="balanced").fit(X_train_final, y_train)
    #scale test set using training fits
    X_test_transformed = scaler.transform(X_test)
    #recombine
    X_test_final = np.concatenate((X_test_transformed,X_test_dummy),axis=1)
    predictions = clf.predict(X_test_final)
    score = clf.score(X_test_final,y_test)
    return predictions, score

def compute_false_negatives(y_test, predictions):
    df = pd.DataFrame({"target": y_test, "predictions": predictions})
    return df[(df["target"] == 1) & (df["predictions"] == 0)].shape[0] / (df[(df["target"] == 1)].shape[0] + 1)

def compute_false_positives(y_test, predictions):
    df = pd.DataFrame({"target": y_test, "predictions": predictions})
    return df[(df["target"] == 0) & (df["predictions"] == 1)].shape[0] / (df[(df["target"] == 0)].shape[0] + 1)


if __name__ == "__main__":
    train, new_dummy_features = read()
    y = get_y(train)
    X = get_X(train,y)
    X_train,X_test,y_train,y_test = split(X,y)
    predictions,score = train_model(X_train,X_test,y_train,y_test,new_dummy_features)
    fn = compute_false_negatives(y_test, predictions)
    fp = compute_false_positives(y_test, predictions)
    print("Accuracy Score: {}".format(score))
    print("False Negatives: {}".format(fn))
    print("False Positives: {}".format(fp))



