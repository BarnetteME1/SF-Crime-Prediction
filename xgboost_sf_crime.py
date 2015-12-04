import scipy
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
le = LabelEncoder()
test = pd.read_csv('SF_crime/test.csv', index_col='Id')
test = test.rename(columns={'X': 'Longitude', "Y": "Latitude"})
test.Dates = pd.to_datetime(test.Dates)
test_keep = test
crime_in_sf = pd.read_csv('SF_crime/train.csv')
crime_in_sf.Dates = pd.to_datetime(crime_in_sf.Dates)
crime_in_sf = crime_in_sf.rename(columns={'X': 'Longitude', "Y": "Latitude",})
crime_in_sf = crime_in_sf.drop(['Resolution', 'Descript'], axis=1)
crime_train, crime_test = train_test_split(crime_in_sf, test_size=.4)

for column in test.columns.values:
    if column != 'Longitude' and column != 'Latitude':
        le.fit(test[column])
        test[column] = le.transform(test[column])

for column in crime_in_sf.columns.values:
    if column != 'Longitude' and column != 'Latitude':
        le.fit(crime_in_sf[column])
        crime_train[column] = le.transform(crime_train[column])

for column in crime_in_sf.columns.values:
    if column != 'Longitude' and column != 'Latitude':
        le.fit(crime_in_sf[column])
        crime_test[column] = le.transform(crime_test[column])

categories = crime_train.Category
crime_train = crime_train.drop('Category', axis=1)

categories2 = crime_test.Category
crime_test = crime_test.drop('Category', axis=1)

dtrain = xgb.DMatrix(crime_train.as_matrix(), label=categories)
dtest = xgb.DMatrix(crime_test.as_matrix(), label=categories2)

param = {'bst:max_depth':6, 'objective':'multi:softprob', 'num_class':39}
param['nthread'] = 4
param['eval_metric'] = ['mlogloss', 'merror']
evallist  = [(dtest,'eval'), (dtrain,'train')]

num_round = 1000
bst = xgb.train(param, dtrain, num_round, evallist, early_stopping_rounds=3)

predictions = bst.predict(xgb.DMatrix(test.as_matrix()), output_margin=False)
predictions = pd.DataFrame(predictions)

le.fit(crime_in_sf.Category)
predictions.columns = le.inverse_transform(predictions.columns)

pred_test = predictions

predictions['Id'] = predictions.index

def order(frame,var):
    varlist =[w for w in frame.columns if w not in var]
    frame = frame[var+varlist]
    return frame
predictions = order(predictions,['Id'])

predictions.to_csv('predictions_XGB_final.csv', index=False)

def dummy_to_column(df):
    columns = df.columns.values
    df_t = df.T
    characters = []
    for column in range(len(columns)):
        for row in range(len(df[columns[column]])):
            if df[columns[column]][row] == '1':
                df[columns[column]][row] = columns[column]
    columns = df_t.columns.values
    characters = []
    for column in range(len(columns)):
        for row in range(len(df_t[columns[column]])):
            if df_t[columns[column]][row] in df_t.index:
                characters.append(df_t[columns[column]][row])
    return characters

def prob_to_column(df):
    df_t = df.T
    columns = df_t.columns.values
    probability = []
    for column in range(len(columns)):
        current_prob = 0
        for row in range(len(df_t[columns[column]])):
            if df_t[columns[column]][row] > current_prob:
                current_prob = df_t[columns[column]][row]
        probability.append(current_prob)
    n = 0
    df_t = df_t.applymap(str)
    for column in range(len(columns)):
        for row in range(len(df_t[column])):
            if df_t[column][row] == str(probability[n]):
                df_t[column][row] = '1'
        n+=1
    return df_t.T

pred_results = pd.DataFrame(dummy_to_column(prob_to_column(pred_test)))
pred_results.to_csv('SF_crime_predictions.csv')

test = test.join(pred_results)
test.to_csv('SF_crime_location.csv')
