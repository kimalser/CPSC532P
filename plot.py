import numpy as np
import matplotlib
import matplotlib.dates as md
import pylab as plt
from datetime import datetime

with open("data/ml-100k/u.data",'r') as ratingsfile:
    all_ratings = (tuple(int(e) for e in line.strip().split('\t'))
                        for line in ratingsfile)
    ratings = [eg for eg in all_ratings if eg[3] <= 884673930]
    all_users = {u for (u,i,r,d) in ratings}
    print("There are ",len(ratings),"ratings and",len(all_users),"users")
    training_users = {u for (u,i,r,d) in ratings if d <= 880845177}
    test_users = all_users - training_users

# extract the training and test dictionaries
with open("data/ml-100k/u.user",'r') as usersfile:
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


# sort by movie and then by date
ratings_sorted = sorted(ratings, key=lambda tup: (tup[1], tup[3]))
plt.figure()

ind = ratings_sorted[0][1] # movie index
start_date = ratings_sorted[0][3]
start_date = datetime.fromtimestamp(start_date)
ave = 0
count = 0
m = [] # an array for each movie; it will contain ratings averaged over 5 calendar days 
for (u,i,r,d) in ratings_sorted:
    d = datetime.fromtimestamp(d)
    if i == ind:
        if (d - start_date).days < 5:
            count += 1
            ave += r
        elif count != 0:
            ave = ave/count
            m.append((start_date, ave))
            ave = 0
            count = 0
            start_date=d
    else:
        x = [a[0] for a in m]
        y = [a[1] for a in m]

        plt.xticks( rotation=25 )
        ax=plt.gca()
        xfmt = md.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(xfmt)
        plt.yticks([1, 2, 3, 4, 5])
        axes = plt.gca()
        axes.set_ylim([0, 5.5])
        plt.plot(x,y)
        title='Movie (id='+str(ind)+') ratings over time'
        print('plotting movie (id=',str(ind),') ratings over time')

        plt.title(title)
        plt.xlabel('Dates')
        plt.ylabel('Ratings')

        filename='plots/movie'+str(ind)+'.png'
        try:
            matplotlib.pyplot.savefig(filename)
        except:
            print('Failed to save the plot. Passing this movie...')
            pass

        plt.close()
        plt.figure()

        # starting new movie
        ind = i
        m = []
        ave = r
        count = 1
        start_date = d
