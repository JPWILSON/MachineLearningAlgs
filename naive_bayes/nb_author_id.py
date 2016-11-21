#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clf = GaussianNB()
t0 = time()
fitting = clf.fit(features_train,labels_train)
train_time = round(time()-t0, 3)
print "training time:", train_time, "s"

t0 = time()
predictions = clf.predict(features_test)
pred_time = round(time()-t0, 3)
print "prediction time:", pred_time, "s"

accuracy = accuracy_score(labels_test, predictions)
print "Accuracy: %s  Training Time: %ss  Prediction Time: %ss" %(accuracy, train_time, pred_time)
#########################################################


'''
t0 = time()
< your clf.fit() line of code >
print "training time:", round(time()-t0, 3), "s"
'''