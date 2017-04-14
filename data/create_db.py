import math, random

datasetname = "100k" # "100k"  # "1m"  # or "Yelp"
if datasetname=="100k":
    datafile, userfilename = "ml-100k/u.data","ml-100k/u.user"
    rating_cutoff, test_cutoff = 884673930, 880845177
elif datasetname=="1m":
    datafile, userfilename = "ml-1m/ratings.dat", "ml-1m/users.dat"
    rating_cutoff, test_cutoff = 974687810, 967587781
elif datasetname=="Yelp":
    datafile, usertrainfilename, usertestfilename = "new MC/yelp_mc_reviews.csv", "new MC/yelp_mc_class_train.csv","new MC/yelp_mc_class_test.csv"
else:
    assert False, ("not a valid dataset name",datasetname)

def extract_cols(lst,indexes):
    """extract sublist given by indexes from lst"""
    return (lst[i] for i in indexes)

with open(datafile,'r') as ratingsfile:
    if datasetname == "Yelp":
        ratings = [(rest,user,int(rating),99999) for line in ratingsfile
                       for (user,rest,rating) in [tuple(line.strip().split(','))]   # for yelp
                             ]
        with open(usertrainfilename) as usertrainfile:
            gender_train = {rest:"F" if eth=="Mexican" else "M" for line in usertrainfile
                                for (rest,eth) in [tuple(line.strip().split(','))]
                                }
        with open(usertestfilename) as usertestfile:
            gender_test = {rest:"F" if eth=="Mexican" else "M" for line in usertestfile
                               for (rest,eth) in [tuple(line.strip().split(','))]
                                }
        training_users = set(gender_train)
        test_users = set(gender_test)
        all_users = training_users | test_users
        #assert all_users == {r for (r,u,i,d) in ratings}
    else:
        if datasetname == "100k":
            all_ratings = (tuple(int(e) for e in line.strip().split('\t'))   # for 100k
                             for line in ratingsfile)
        elif datasetname == "1m":
            all_ratings = (tuple(int(e) for e in extract_cols(line.strip().split(':'),[0,2,4,6])) # for 1m
                         for line in ratingsfile)
        ratings = [eg for eg in all_ratings if eg[3] <= rating_cutoff]
        all_users = {u for (u,i,r,d) in ratings}
        print("There are ",len(ratings),"ratings and",len(all_users),"users")
        training_users = {u for (u,i,r,d) in ratings if d <= test_cutoff}
        test_users = all_users - training_users

        # extract the training and test dictionaries
        with open(userfilename,'r') as usersfile:
            if datasetname == "100k":
                user_info = (line.strip().split('|') for line in usersfile)
            elif datasetname == "1m":
                user_info = (extract_cols(line.strip().split(':'),[0,4,2,6,8]) for line in usersfile)
            gender_train, gender_test = {},{}
            for (u,a,g,o,p) in user_info:
                if int(u) in training_users:
                    gender_train[int(u)]=g
                elif int(u) in test_users:
                    gender_test[int(u)]=g

# check the results
assert len(training_users)==len(gender_train),(len(training_users),len(gender_train))
assert len(test_users)==len(gender_test), (len(test_users),len(gender_test))
print("There are ",len(gender_train),"users for training")
print("There are ",len(gender_test),"users for testing")

# import numpy as np
# print(len(ratings))
# mylist = list(set(ratings))
# print(len(mylist))


user_mov_rat_gen = list()
for (u,i,r,d) in ratings:
  if u in gender_test:   
    if gender_test[u]=="F":
        user_mov_rat_gen.append((u,i,r,1))
    if gender_test[u]=="M":
        user_mov_rat_gen.append((u,i,r,0))
user_mov_rat_gen.sort(key=lambda tup: tup[0])

f = open('../db/users_train_100k_rated.db', 'w')
for u in gender_test:
	f.write('user('+str(u)+')\n')
	
for i in range(len(user_mov_rat_gen)):
    f.write('rated'+'('+str(user_mov_rat_gen[i][0]) + ',' + str(user_mov_rat_gen[i][1])+')\n')

    # if user_mov_rat_gen[i][2] >= 4:
    #     f.write('rating_gr_eq_4'+'('+str(user_mov_rat_gen[i][0]) + ',' + str(user_mov_rat_gen[i][1])+')\n')
    # else:
    #     f.write('rating_less_4'+'('+str(user_mov_rat_gen[i][0]) + ',' + str(user_mov_rat_gen[i][1])+')\n')

for u in gender_train:
    if gender_train[u]=="F":
        f.write('female('+str(u)+')\n')
    if gender_train[u]=="M":
        f.write('!female('+str(u)+')\n')
