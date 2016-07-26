# coding=utf8
#eval-mlogloss:2.30607
#public 2.30336

#This script is an adaptation from the bag of apps script published in the forums.
#In this script I added the event_count.csv dataframe.
#If you are low on RAM uncomment the 'del' lines.
#If you have any problems with this get in touch.
#Maybe some imports can be cleaned I have already messed quite a bit with this script.
#Bernardo Doré

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import random

random.seed(2016)

# Create bag-of-apps in character string format
# first by event
# then merge to generate larger bags by device

##################
#   App Labels
##################

print("# Read App Labels")
app_lab = pd.read_csv("../input/app_labels.csv", dtype={'device_id': np.str})
app_lab = app_lab.groupby("app_id")["label_id"].apply(
    lambda x: " ".join(str(s) for s in x))

##################
#   App Events
##################
print("# Read App Events")
app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str})
app_ev["app_lab"] = app_ev["app_id"].map(app_lab)
app_ev = app_ev.groupby("event_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

#del app_lab

##################
#     Events
##################
print("# Read Events")
events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})
events["app_lab"] = events["event_id"].map(app_ev)
events = events.groupby("device_id")["app_lab"].apply(
    lambda x: " ".join(str(s) for s in x))

#del app_ev

##################
#   Phone Brand
##################
print("# Read Phone Brand")
pbd = pd.read_csv("../input/phone_brand_device_model.csv",
                  dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)


##################
#  Train and Test
##################
print("# Generate Train and Test")

#Read even_count.csv
event_counts = pd.read_csv("event_count.csv")
event_counts = event_counts.drop(['device_id.1'], axis=1)

train = pd.read_csv("../input/gender_age_train.csv",
                    dtype={'device_id': np.str})
train["app_lab"] = train["device_id"].map(events)

#merge even_count.csv
train = pd.merge(train, event_counts, how='left', on='device_id', left_index=True)
train = pd.merge(train, pbd, how='left',
                 on='device_id', left_index=True)

test = pd.read_csv("../input/gender_age_test.csv",
                   dtype={'device_id': np.str})
test["app_lab"] = test["device_id"].map(events)

#merge even_count.csv
test = pd.merge(test, event_counts, how='left', on='device_id', left_index=True)
test = pd.merge(test, pbd, how='left',
                on='device_id', left_index=True)

##################
#   Build Model
##################

#The "problem" with this model is that it uses feature hashing and doing so transforms the event_count numerical data into string data.
#All data is treated as a frequency of string terms which makes some sense in the context of usage frequency but I think it's not ideal
#since the nature of counts is numerical and we want the data to be treated as such.
#I tried bulding a sparse matrix from event_count and adding it to the matrices resulting of get_hash_data with worse results. I had
#to introduce too much 0's to satisfy the matrice's shape.

hash_list = list(train.columns)
remove_from_hash = ['device_id','gender','age','group']
hash_list = [x for x in hash_list if x not in remove_from_hash]


def get_hash_data(train, test):
    df = pd.concat((train, test), axis=0, ignore_index=True)
    split_len = len(train)

    # TF-IDF Feature    
    tfv = CountVectorizer(min_df=1, binary=1)
    df = df[hash_list].astype(np.str).apply(
        lambda x: " ".join(s for s in x), axis=1).fillna("Missing")
    df_tfv = tfv.fit_transform(df)

    train = df_tfv[:split_len, :]
    test = df_tfv[split_len:, :]
    return train, test

# Group Labels
Y = train["group"]
lable_group = LabelEncoder()
Y = lable_group.fit_transform(Y)

device_id = test["device_id"].values
train, test = get_hash_data(train,test)

X_train, X_val, y_train, y_val = train_test_split(train, Y, train_size=.80)


##################
#     XGBoost
##################

dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_val, y_val)

params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gbtree",
    "eval_metric": "mlogloss",
    "eta": 0.1,
    "silent": 1,
}
watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 366, evals=watchlist, verbose_eval=True)

y_pre = gbm.predict(xgb.DMatrix(test), ntree_limit=gbm.best_iteration)

# Write results
result = pd.DataFrame(y_pre, columns=lable_group.classes_)
result["device_id"] = device_id
result = result.set_index("device_id")
result.to_csv('test_Result.csv', index=True, index_label='device_id')