import pandas as pd
import math
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def main():
    jfkLong = -73.778889
    jfkLat = 40.639722
    lgLong = -73.872611
    lgLat = 40.77725
    pd.options.mode.chained_assignment = None  # default='warn'
    train = pd.read_csv('C:/Users/Philip/PycharmProjects/NYCTaxi/train.csv')
    test = pd.read_csv('C:/Users/Philip/PycharmProjects/NYCTaxi/test.csv')


    train['pickup_datetime'] = pd.to_datetime(train['pickup_datetime'])
    train['dropoff_datetime'] = pd.to_datetime(train['dropoff_datetime'])
    test['pickup_datetime'] = pd.to_datetime(test['pickup_datetime'])

    train.loc[train.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 0
    train.loc[train.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 1
    test.loc[test.store_and_fwd_flag == 'N', 'store_and_fwd_flag'] = 0
    test.loc[test.store_and_fwd_flag == 'Y', 'store_and_fwd_flag'] = 1
    train.store_and_fwd_flag = pd.to_numeric(train.store_and_fwd_flag)
    test.store_and_fwd_flag = pd.to_numeric(test.store_and_fwd_flag)


    airportDist = 2
    train['dayofweek'] = [n.dayofweek for n in train['pickup_datetime']]
    train['hour'] = [n.hour for n in train.pickup_datetime]
    train['weeknumber'] = [n.week for n in train['pickup_datetime']]
    train['hourofweek'] = (train.dayofweek * 24) + train.hour
    train['dist'] = [distBetween(train.loc[i,'pickup_longitude'],train.loc[i,'pickup_latitude'],train.loc[i,'dropoff_longitude'],train.loc[i,'dropoff_latitude']) for i in range(len(train.index))]
    train['jfkTrip'] = [isAirport(train.loc[i,'pickup_longitude'],train.loc[i,'pickup_latitude'],train.loc[i,'dropoff_longitude'],train.loc[i,'dropoff_latitude'],jfkLong,jfkLat,airportDist) for i in range(len(train.index))]
    train['lgTrip'] = [isAirport(train.loc[i,'pickup_longitude'],train.loc[i,'pickup_latitude'],train.loc[i,'dropoff_longitude'],train.loc[i,'dropoff_latitude'],lgLong,lgLat,airportDist) for i in range(len(train.index))]

    test['dayofweek'] = [n.dayofweek for n in test['pickup_datetime']]
    test['hour'] = [n.hour for n in test.pickup_datetime]
    test['weeknumber'] = [n.week for n in test['pickup_datetime']]
    test['hourofweek'] = (test.dayofweek * 24) + test.hour
    test['dist'] = [distBetween(test.loc[i, 'pickup_longitude'], test.loc[i, 'pickup_latitude'],
                                 test.loc[i, 'dropoff_longitude'], test.loc[i, 'dropoff_latitude']) for i in
                     range(len(test.index))]
    test['jfkTrip'] = [
        isAirport(test.loc[i, 'pickup_longitude'], test.loc[i, 'pickup_latitude'], test.loc[i, 'dropoff_longitude'],
                  test.loc[i, 'dropoff_latitude'], jfkLong, jfkLat, airportDist) for i in range(len(test.index))]
    test['lgTrip'] = [
        isAirport(test.loc[i, 'pickup_longitude'], test.loc[i, 'pickup_latitude'], test.loc[i, 'dropoff_longitude'],
                  test.loc[i, 'dropoff_latitude'], lgLong, lgLat, airportDist) for i in range(len(test.index))]

    train.set_index('id', inplace=True)
    test.set_index('id', inplace=True)

    X, y = XYSplit(train)
    del X['dropoff_datetime']
    del X['pickup_datetime']
    del test['pickup_datetime']


    #Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.33, random_state=0)

    print("training...")
    rfr_log = RandomForestRegressor(random_state=0).fit(X, np.log(y))
    print(X.columns)
    print(rfr_log.feature_importances_)
    pred_rf_log = np.e ** (rfr_log.predict(test))
    my_solution = pd.DataFrame(pred_rf_log, test['id'], columns=["trip_duration"])
    my_solution.to_csv("out6.csv", index_label=["id"])

    # rmsle_rf = rmsle(ytest, pred_rf_log)
    # print(rmsle_rf)





def distBetween(long1, lat1, long2, lat2):
    R = 6373 #Earth's radius
    dlon = math.radians(long2 - long1)
    dlat = math.radians(lat2 - lat1)
    a = (math.sin(dlat / 2)) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * (math.sin(
        dlon / 2)) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def XYSplit(train):
    X = train.copy()
    y = train.trip_duration
    del X['trip_duration']
    return X, y

def rmsle(real,predicted):
    sum=0.0
    for x in range(len(predicted)):
        p = np.log(predicted[x])
        r = np.log(real[x])
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5

def isAirport(longitude1,latitude1,longitude2,latitude2,aLong,aLat,airportDist):
    if(distBetween(longitude1,latitude1,aLong,aLat) < airportDist or distBetween(longitude2,latitude2,aLong,aLat) < airportDist):
        return 1
    return 0


if __name__ == '__main__':
        main()