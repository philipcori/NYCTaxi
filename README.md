# NYCTaxi

Project that predicts the duration of a NYC taxi trip given certain features of the trip, from the competition on kaggle.com: New York
City Taxi Trip Duration. Both main.py and main2.py use train a Random Forest Regresor model using a training dataset. An output
file is then constructed, filled with predictions produced from a test.csv file. 

I used the pandas library to read and organize the datasets, taking advantage of its useful DataFrame object. The scikit-learn library was 
used for its machine learning models, in this case the Random Forest Regresor. Lastly, the matplotlib library was used to visualize certain
parts of the data.

main.py is a first attempt that utilized feature engineering, data cleaning, and a RandomForestRegresor object from the scikit-learn
library. I also used matplotlib to graph things like time vs distance travelled and time vs other features to look into any possible 
correlations.

main2.py is a streamlined version of main.py, which improves and adds on to some of the feature engineering done in main.py, performs a logarithmic transformation on the data to normalize the dataset which improved accuracy, and also improves runtime.

The optimal program produced a Root Mean Squared Logarithmic Error of 0.42503.
