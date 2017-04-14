# # CPSC 532P 2017 - Predicting Gender from Ratings
# Copyright Alexandra Kim 2017. You may use it under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See: http://creativecommons.org/licenses/by-nc-sa/4.0/deed.en
# Dataset loading and evaluation section are adapted from D. Poole.

# ## Load the files. 
# To use this you need to download http://files.grouplens.org/datasets/movielens/ml-100k.zip
# See http://grouplens.org/datasets/movielens/
# 
# The following reads the ratings file and selects temporally first 60000 ratings.
# It trains on the users who were involved in first 40000 ratings.
# It tests on the other users who rated.

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
import math


with open("ml-100k/u.data",'r') as ratingsfile:
    all_ratings = (tuple(int(e) for e in line.strip().split('\t'))
                        for line in ratingsfile)
    ratings = [eg for eg in all_ratings if eg[3] <= 884673930]
    all_users = {u for (u,i,r,d) in ratings}
    print("There are ",len(ratings),"ratings and",len(all_users),"users")
    training_users = {u for (u,i,r,d) in ratings if d <= 880845177}
    test_users = all_users - training_users

# extract the training and test dictionaries
with open("ml-100k/u.user",'r') as usersfile:
    user_info = (line.strip().split('|') for line in usersfile)
    gender_train, gender_test = {},{}
    for (u,a,g,o,p) in user_info:
        if int(u) in training_users:
            gender_train[int(u)]=g
        elif int(u) in test_users:
            gender_test[int(u)]=g

# check the results
assert len(training_users)==len(gender_train)
assert len(test_users)==len(gender_test)
print("There are ",len(gender_train),"users for training")
print("There are ",len(gender_test),"users for testing")


user_to_movie_and_rating = {} # user -> [(movie,rating)] dictionary
for (u,i,r,d) in ratings:
    if u not in user_to_movie_and_rating:
        user_to_movie_and_rating[u] = []
    user_to_movie_and_rating[u].append((i,r))



def evaluate(pred,para=1):
    """pred is a function from users into real numbers that gives prediction P(u)='F',
    returns (sum_squares_error,  log loss)"""
    sse = sum((pred(u,para=para)-(1 if g=="F" else 0))**2
                  for (u,g) in gender_test.items())
    ll = -sum(math.log(pr,2) if g=='F' else math.log(1-pr,2)
                  for (u,g) in gender_test.items()
                  for pr in [pred(u,para=para)])
    return (sse,ll)



# Building a matrix for classification
# ---------------------------------------
movies = [x[1] for x in ratings]

# lists of unique users and movies
userList = list(training_users)
movieList = list(set(movies))

# mapping from user/movie ID to a position in the matrix
userMapping = {} 
movieMapping = {}

for i in range(len(userList)):
    userMapping[userList[i]] = i;

for i in range(len(movieList)):
    movieMapping[movieList[i]] = i;

M = np.zeros((len(userMapping), len(movieMapping)))

for (u,i,r,d) in ratings:
    if u in gender_train:
        M[userMapping[u], movieMapping[i]] = r

# build a corresponding results matrix
gen = np.zeros(len(userMapping))
for u, g in gender_train.iteritems():
    if g == 'F':
        gen[userMapping[u]] = 1
    if g == 'M':
        gen[userMapping[u]] = 0

# binary matrix where 1 indicates that a user rated a movie
M_rated = np.copy(M)
M_rated[M_rated > 0] = 1

# binary matrix where 1 indicates that a user gave a movie a rating of 4 or 5 stars
M_rating = np.copy(M)
M_rating[M_rating < 4] = 0 
M_rating[M_rating >= 4] = 1
# ---------------------------------------


# SVM classifier
# ---------------------------------------
svm_rated = svm.SVC(probability=True)
svm_rated.fit(M_rated, gen)
svm_rating = svm.SVC(probability=True)
svm_rating.fit(M_rating, gen)  


def pred_svm_rated(u,para=1):
    # movie vector for a given user
    u_movies = np.zeros(len(movieList)) 

    for m,r in user_to_movie_and_rating[u]:
        u_movies[movieMapping[m]] = 1

    u_movies = u_movies.reshape(1, -1)
    return svm_rated.predict_proba(u_movies)[0][1]


def pred_svm_rating(u,para=1):
    # movie vector for a given user
    u_movies = np.zeros(len(movieList)) 

    for m,r in user_to_movie_and_rating[u]:
        if r >= 4:
            u_movies[movieMapping[m]] = 1

    u_movies = u_movies.reshape(1, -1)
    return svm_rating.predict_proba(u_movies)[0][1]


print("Errors for SVM (rated)", evaluate(pred_svm_rated))
print("Errors for SVM (rating)", evaluate(pred_svm_rating))
# ---------------------------------------



# Logistic regression
# ---------------------------------------
logreg_rated = linear_model.LogisticRegression(class_weight='balanced' , max_iter=1000)
logreg_rated.fit(M_rated, gen)
logreg_rating = linear_model.LogisticRegression(class_weight='balanced' , max_iter=1000)
logreg_rating.fit(M_rating, gen)

def pred_logreg_rated(u,para=1):
    # movie vector for a given user
    u_movies = np.zeros(len(movieList)) 

    for m,r in user_to_movie_and_rating[u]:
        u_movies[movieMapping[m]] = 1
    u_movies = u_movies.reshape(1, -1)
 
    return logreg_rated.predict_proba(u_movies)[0][1]


def pred_logreg_rating(u,para=1):
    # movie vector for a given user
    u_movies = np.zeros(len(movieList)) 

    for m,r in user_to_movie_and_rating[u]:
        u_movies[movieMapping[m]] = 1
    u_movies = u_movies.reshape(1, -1)
 
    return logreg_rating.predict_proba(u_movies)[0][1]


print("Errors for logistic regression (rated)", evaluate(pred_logreg_rated))
print("Errors for logistic regression (rating)", evaluate(pred_logreg_rating))
# ---------------------------------------



# Neural network
# ---------------------------------------
nn_rated = MLPClassifier(solver='sgd', max_iter=200)
nn_rated.fit(M_rated, gen)
nn_rating = MLPClassifier(solver='sgd', max_iter=200)
nn_rating.fit(M_rated, gen)

def pred_nn_rated(u,para=1):
    # movie vector for a given user
    u_movies = np.zeros(len(movieList)) 

    for m,r in user_to_movie_and_rating[u]:
        u_movies[movieMapping[m]] = 1
    u_movies = u_movies.reshape(1, -1)
 
    return nn_rated.predict_proba(u_movies)[0][1]

def pred_nn_rating(u,para=1):
    # movie vector for a given user
    u_movies = np.zeros(len(movieList)) 

    for m,r in user_to_movie_and_rating[u]:
        u_movies[movieMapping[m]] = 1
    u_movies = u_movies.reshape(1, -1)
 
    return nn_rating.predict_proba(u_movies)[0][1]


print("Errors for neural network (rated)", evaluate(pred_nn_rated))
print("Errors for neural network (rating)", evaluate(pred_nn_rating))

