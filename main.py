import pandas as pd
import numpy as np
import sys
import datetime
import sklearn
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression

def main():

    pd.options.mode.chained_assignment = None  # default='warn'

    train = pd.read_csv('C:/Users/Philip/PycharmProjects/NYCTaxi/train.csv')
    test = pd.read_csv('C:/Users/Philip/PycharmProjects/NYCTaxi/test.csv')

    jfkLong = -73.778889
    jfkLat = 40.639722
    lgLong = -73.872611
    lgLat = 40.77725

    y = train["trip_duration"].values
    pLong = train["pickup_longitude"].values
    pLat = train["pickup_latitude"].values
    dLong = train["dropoff_longitude"].values
    dLat = train["dropoff_latitude"].values
    pickup_datetime = train["pickup_datetime"].values

    # hour = np.ndarray(shape=(len(y)),dtype=int)
    # minute = np.ndarray(shape=(len(y)),dtype=int)

    dist = np.ndarray(shape=(len(y),1),dtype=float)
    avgSpeed = np.ndarray(shape=(len(y),1),dtype=float)
    airportDist = 2


    jfkTrip = np.ndarray(shape=(len(y),1),dtype=float)
    lgTrip = np.ndarray(shape=(len(y),1),dtype=float)
    weekDay = np.ndarray(shape=(len(y),1),dtype=float)
    hour = np.ndarray(shape=(len(y),1),dtype=float)
    work = np.ndarray(shape=(len(y),1),dtype=float)


    for i in range(len(train.index)):
        dist[i] = (distBetween(pLong[i], pLat[i], dLong[i], dLat[i]))
        avgSpeed[i] = dist[i] / (y[i] / 3600)
        hour[i] = float(pickup_datetime[i][11:13]) #changes to hour
        month = int(pickup_datetime[i][5:7])
        day = int(pickup_datetime[i][8:10])
        date = datetime.date(2016,month=month,day=day)
        weekDay[i] = date.weekday()
        if(hour[i] > 8 and hour[i] < 18 and weekDay[i] < 5):
            work[i] = 1
        else:
            work[i] = 0
        if(distBetween(jfkLong,jfkLat,pLong[i],pLat[i]) < airportDist or distBetween(jfkLong,jfkLat,dLong[i],dLat[i]) < airportDist):
           # jfkDurations.append(y[i])
            jfkTrip[i] = 1
        else:
            jfkTrip[i] =0
        if (distBetween(lgLong, lgLat, pLong[i], pLat[i]) < airportDist or distBetween(lgLong, lgLat, dLong[i],dLat[i]) < airportDist):
            # lgDurations.append(y[i])
             lgTrip[i] = 1
        else:
            lgTrip[i] = 0
        if (i % 100000 == 0):
            print(i)


    train["dist"] = dist
    train["avgSpeed"] = avgSpeed
    train["hour"] = hour
    train["weekDay"] = weekDay
    train["jfkTrip"] = jfkTrip
    train["lgTrip"] = lgTrip
    train["work"] = work

    print(train["jfkTrip"].describe())
    print(train["lgTrip"].describe())

   # cleanData(train)

    # jfkDurations = np.array(jfkDurations)
    # lgDurations = np.array(lgDurations)
#    print(len(jfkDurations) + len(lgDurations))
    #jfkMean = jfkDurations.mean() - train["trip_duration"].mean()
    #lgMean = lgDurations.mean() - train["trip_duration"].mean()

    target = train["trip_duration"].values
    features = train[["jfkTrip","lgTrip","vendor_id","passenger_count","dist","hour","weekDay","work"]].values
    forest = RandomForestRegressor(n_estimators=100,max_depth=4,min_samples_split=2, n_jobs=1)
    print("training...")
    my_forest = forest.fit(features,target)

    print(my_forest.feature_importances_)

   # sys.exit()

    # model = LinearRegression()
    # model.fit(dist,y)

    pLong2 = test["pickup_longitude"].values
    pLat2 = test["pickup_latitude"].values
    dLong2 = test["dropoff_longitude"].values
    dLat2 = test["dropoff_latitude"].values
    pickup_datetime2 = test["pickup_datetime"].values

    prediction = np.ndarray(shape=(len(test.index)),dtype=float)

    dist2 = np.ndarray(shape=(len(prediction)),dtype=float)
    jfkTrip2 = np.ndarray(shape=(len(prediction)),dtype=float)
    lgTrip2 = np.ndarray(shape=(len(prediction)),dtype=float)
    weekDay2 = np.ndarray(shape=(len(prediction), 1), dtype=float)
    hour2 = np.ndarray(shape=(len(prediction), 1), dtype=float)
    work2 = np.ndarray(shape=(len(prediction), 1), dtype=float)


  #  for i in range(len(prediction)):
        # hour = int(pickup_datetime2[i][11:13])
        # minute = int(pickup_datetime2[i][14:16]) / 60
        #pickup_datetime2[i] = hour + minute


    for i in range(len(prediction)):
        dist2[i] = distBetween(pLong2[i],pLat2[i],dLong2[i],dLat2[i])
        hour2[i] = float(pickup_datetime2[i][11:13])  # changes to hour
        month = int(pickup_datetime2[i][5:7])
        day = int(pickup_datetime2[i][8:10])
        date = datetime.date(2016, month=month, day=day)
        weekDay2[i] = date.weekday()
        if (hour2[i] > 8 and hour2[i] < 18 and weekDay2[i] < 5):
            work2[i] = 1
        else:
            work2[i] = 0
        if (distBetween(jfkLong, jfkLat, pLong2[i], pLat2[i]) < airportDist or distBetween(jfkLong, jfkLat, dLong2[i], dLat2[i]) < airportDist):
            jfkTrip2[i] = 1
        else:
            jfkTrip2[i] = 0
        if (distBetween(lgLong, lgLat, pLong2[i], pLat2[i]) < airportDist or distBetween(lgLong, lgLat, dLong2[i], dLat2[i]) < airportDist):
            lgTrip2[i] = 1
        else:
            lgTrip2[i] = 0

    test["jfkTrip"] = jfkTrip2
    test["lgTrip"] = lgTrip2
    test["dist"] = dist2
    test["hour"] = hour2
    test["work"] = work2
    test["weekDay"] = weekDay2

    test_features = test[["jfkTrip","lgTrip","vendor_id","passenger_count","dist","hour","weekDay","work"]].values
    pred_forest = my_forest.predict(test_features)

    id = test["id"]
    my_solution = pd.DataFrame(pred_forest, id, columns=["trip_duration"])
    my_solution.to_csv("out5.csv",index_label=["id"])

    # axes = plt.gca()
    # axes.set_xlim([0,20])
    # axes.set_ylim([0,6000])
    # plt.plot(dist,y,'.')
    # plt.plot(dist2,prediction,'r.')
    # plt.show()




def distBetween(long1, lat1, long2, lat2):
    R = 6373 #Earth's radius
    dlon = math.radians(long2 - long1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2)) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(
        dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def cleanData(train):
    filter = []
    len1 = len(train.index)
    for i in range(len1):
        if(train["trip_duration"][i] < 10 or train["trip_duration"][i] > 22*3600):
            filter.append(False)
        elif(train["dist"][i] == 0 and train["trip_duration"][i] > 60):
            filter.append(False)
        elif(train["avgSpeed"][i] > 100):
            filter.append(False)
        else:
            filter.append(True)

    filter = pd.Series(filter)
    train = train[filter]

    percent = (len1 - len(train.index))/len1 * 100
    print("Cleaned out %f percent" %(percent))


if __name__ == '__main__':
        main()