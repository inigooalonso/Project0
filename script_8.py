# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 01:25:43 2016

Based on yibo's R script and JianXiao's translation to Python

@author: Tony
"""
#LB: 

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.decomposition import PCA, RandomizedPCA, SparsePCA, TruncatedSVD
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss
from sklearn.preprocessing import MaxAbsScaler

###########################################################
# Concatenate labels and categories to attach to each app #
###########################################################
app_labels = pd.read_csv("../input/app_labels.csv", dtype={'app_id': np.str})
label_categories = pd.read_csv("../input/label_categories.csv")
label_categories.dropna(inplace=True)
app_lab_cat = pd.merge(app_labels, label_categories, on='label_id', how='left')

del label_categories
del app_labels

app_lab_cat['label_id'] = app_lab_cat['label_id'].map(lambda x: "l:"+str(x))
app_lab_cat['category'] = app_lab_cat['category'].map(
    lambda x: " ".join("c:"+ str(c) for c in x.lower().replace('-', "").replace(" /", "").split(' ')))

app_lab = app_lab_cat.groupby('app_id')['label_id'].apply(
    lambda x: " ".join(str(feature) for feature in x))
app_cat = app_lab_cat.groupby('app_id')['category'].apply(
    lambda x: " ".join(str(feature) for feature in x))
    
del app_lab_cat

##################################################
# Merge features onto apps in events/device_id's #
##################################################
app_ev = pd.read_csv("../input/app_events.csv", dtype={'device_id': np.str, 'app_id': np.str})
app_ev.drop(['is_installed', 'is_active'], axis=1, inplace=True)
app_ev['app_lab'] = app_ev['app_id'].map(app_lab)
app_ev['app_cat'] = app_ev['app_id'].map(app_cat)

del app_lab
del app_cat

events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str, 'app_id': np.str})
events.drop(['timestamp', 'longitude', 'latitude'], axis=1, inplace=True)
events = events[['device_id', 'event_id']]
events = pd.merge(events, app_ev, on='event_id', how='left')

del app_ev

events.dropna(inplace=True)
events.drop('event_id', axis=1, inplace=True)
events.drop_duplicates(['device_id', 'app_id'], inplace=True)

#############################################################
# Merge features and prep for training with CountVectorizer #
#############################################################

events_app = events.groupby("device_id")['app_id'].apply(lambda x: " ".join(s for s in x))
events_lab = events.groupby("device_id")['app_lab'].apply(lambda x: " ".join(s for s in x))
events_cat = events.groupby("device_id")['app_cat'].apply(lambda x: " ".join(s for s in x))

events = pd.DataFrame({'app_id':events_app, 'labels':events_lab, 'categories':events_cat})

del events_app
del events_lab
del events_cat

pbd = pd.read_csv("../input/phone_brand_device_model.csv", dtype={'device_id': np.str})
pbd.drop_duplicates('device_id', keep='first', inplace=True)
#pbd['device_model'] = pbd['phone_brand'] + pbd['device_model']
pbd = pd.merge(pbd, events, left_on='device_id', right_index=True, how='left')
pbd.fillna("", inplace=True)

brands_models = pbd[['device_id','phone_brand', 'device_model']]
pbd.drop(['phone_brand', 'device_model'], axis=1, inplace=True)

del events

##################
#  Train and Test
##################

train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})
y = train["group"]
le = LabelEncoder()
y = le.fit_transform(y)
train.drop(['age', 'gender', 'group'], axis=1, inplace=True)
train_len = len(train)

test = pd.read_csv("../input/gender_age_test.csv", dtype={'device_id': np.str})
test_dev = test["device_id"]

# Concat
df = pd.concat((train, test), axis=0, ignore_index=True)

# Create brand and model Series in order to label encode them
brands_models = pd.merge(df, brands_models, how='left', on='device_id')
brands = brands_models['phone_brand']
models = brands_models['device_model']
del brands_models

# Create feature df with all feature except phone_brand and device_model
df = pd.merge(df, pbd, how='left', on='device_id')
del pbd
dev_app = df['app_id']
dev_cat = df['categories']
dev_lab = df['labels']

# For each device, count app_id's, labels, and category names
app_vectorizer = CountVectorizer(binary=True)
dev_app = app_vectorizer.fit_transform(dev_app)

lab_vectorizer = CountVectorizer(binary=True)
dev_lab = lab_vectorizer.fit_transform(dev_lab)

cat_vectorizer = CountVectorizer(binary=True)
dev_cat = cat_vectorizer.fit_transform(dev_cat)

cv_brand = CountVectorizer(binary=True)
brands = cv_brand.fit_transform(brands)

cv_model = CountVectorizer(binary=True)
models = cv_model.fit_transform(models)

X_sp = sparse.hstack([brands, dev_app, dev_lab, models], format='csr')
del models
del brands
del dev_lab
del dev_cat
del dev_app

# Scale features
#scaler = MaxAbsScaler()
#X_sp = scaler.fit_transform(X_sp)

# Split X into train/cv/test
X_train_cv = X_sp[:train_len,:]
X_test = X_sp[train_len:,:]

X_train, X_cv, y_train, y_cv = train_test_split(
    X_train_cv, y, train_size=.90, random_state=10)

#sss = StratifiedShuffleSplit(y, train_size=.9, random_state=10)
#for train_index, cv_index in sss:
#    X_train, X_cv = X_train_cv[train_index,:], X_train_cv[cv_index,:]
#    y_train, y_cv = y[train_index], y[cv_index]

##################
#   Feature Sel
##################
#print("# Feature Selection")
#selector = TruncatedSVD(n_components=100, random_state=1)
#
#selector.fit(X_sp)
#
#X_train = selector.transform(X_train)
#X_cv = selector.transform(X_cv)
#
#X_test = selector.transform(X_test)
#
#print("# Num of Features: ", X_train.shape[1])

##################
#  Build Models
##################
num_inputs = X_train.shape[1]
n_train = X_train.shape[0]
hidden_units_1 = 48
num_classes = 12
p_dropout = 0.0

def indicator(y):
    ind = np.zeros((y.shape[0], num_classes))
    ind[np.arange(y.shape[0]), y] = 1
    return ind

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
    # create model
    model = Sequential()
    model.add(Dense(output_dim=64, input_dim=num_inputs, W_regularizer=l2(1.0)))
    model.add(Activation("relu"))
    model.add(Dense(output_dim=num_classes, W_regularizer=l2(1.0)))
    model.add(Activation("softmax"))
    # Compile model
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])
    return model

class KerasClassifier2(KerasClassifier):
        
    def __init__(self, build_fn, random_state=0, nb_epoch=6, batch_size=500, verbose=2):
        self.random_state = random_state
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        super(KerasClassifier2, self).__init__(build_fn)
        self.classes_= np.arange(12)
        self.n_classes_ = 12
        
    def fit(self, X, y, sample_weight=None):
        return super(KerasClassifier2, self).fit(X, indicator(y), verbose = self.verbose,
                                sample_weight=sample_weight*n_train,
                                validation_data=(X_cv, indicator(y_cv)),
                                nb_epoch=self.nb_epoch, batch_size=self.batch_size)
    def predict_proba(self, X):
        return super(KerasClassifier2, self).predict_proba(X, batch_size=500, verbose=0)

class classifier_booster(object):
    '''
    Takes in a classifier and creates boosted classifier similar to Adaboost
    but trains on log loss instead of exponential loss. In creation, clf is the
    base, n_estimators is how many rounds of boosting are completed and pct is
    what percentage of data points are used in each round to do the boost
    '''
    def __init__(self, clf, n_estimators, pct):
        self.clf = clf
        self.n_estimators = n_estimators
        self.pct = pct
        
    def indicator(y):
        ind = np.zeros((y.shape[0], num_classes))
        ind[np.arange(y.shape[0]), y] = 1
        return ind
        
    def fit(self, X, y, sample_weight = None):
        n_samples = X.shape[0]
        if sample_weight == None:
            sample_weight = np.ones(n_samples)/float(n_samples)
        models = []
        model_weights = []
        for i in xrange(self.n_estimators):
            self.clf.fit(X, y, sample_weight)
            pred_prob = self.clf.predict_proba(X)
            pred_prob[pred_prob < 1e-15] = 1e-15
            pred_prob[pred_prob > 1-1e-15] = 1-1e-15
            err = -np.multiply(np.log(pred_prob), indicator(y)).sum(axis=1)
            
    def predict_proba(self, X):
        model.pred
        
        

clfNN = KerasClassifier2(build_fn=create_model)
#clfAdaNN = AdaBoostClassifier(base_estimator=clfNN, n_estimators=5, learning_rate=1)
wgts = np.ones(n_train)/float(n_train)
clfNN.fit(X_train, y_train, sample_weight=wgts)
clfAdaNN.fit(X_train, y_train)


def logloss(pred_prob, actual):
    # Takes the probabilities of each class and compares them with actual
    # values to give the log loss. Limit values near 0 and 1 since they are
    # undefined. Returns log loss only of actual class.
    pred_prob[pred_prob < 1e-15] = 1e-15
    pred_prob[pred_prob > 1-1e-15] = 1-1e-15
    log_prob = np.log(pred_prob)
    indicator_actual = np.zeros(pred_prob.shape)
    indicator_actual[np.arange(len(actual)), actual] = 1
    err = -np.multiply(log_prob, indicator_actual)
    return err.sum()/float(err.shape[0])
#print("Log Loss = {}".format(logloss(pred_prob_cv_nn, y_cv)))
### Log Loss = 


dtrain = xgb.DMatrix(X_train, y_train)
dvalid = xgb.DMatrix(X_cv, y_cv)

params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gblinear",
    "max_depth":6,
    "eval_metric": "mlogloss",
    "eta": 0.07,
    "silent": 1,
    "alpha":3.5,
}

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]
gbm = xgb.train(params, dtrain, 40, evals=watchlist,
                early_stopping_rounds=25, verbose_eval=True)

print("# Train")
dtrain = xgb.DMatrix(X_train, y_train)
gbm = xgb.train(params, dtrain, 40, verbose_eval=True)
y_pre = gbm.predict(xgb.DMatrix(X_test))
### cv log_loss=2.30242

clfLR = LogisticRegression(C=.02, random_state=2016, multi_class='multinomial', solver='lbfgs')
clfLR.fit(X_train,y_train)
print  "Log Loss = {}".format(logloss(clfLR.predict_proba(X_cv), y_cv))
pred_prob_LR = clfLR.predict(X_test)

#clfMLP = MLPClassifier()
#clfMLP.fit(X_train,y_train)
#print  "Log Loss = {}".format(logloss(clfMLP.predict_proba(X_cv), y_cv))
#pred_prob_MLP = clfMLP.predict(X_test)

#Combine results of NN and xgboost
#y_pre = (pred_prob_nn+y_pre)/2.0 

# Write results
result = pd.DataFrame(y_pre, columns=le.classes_)
result["device_id"] = test_dev
result = result.set_index("device_id")
result.to_csv('counting_features.gz', index=True,
              index_label='device_id', compression="gzip")
              
#result = pd.DataFrame(pred_prob_LR, columns=le.classes_)
#result["device_id"] = test_dev
#result = result.set_index("device_id")
#result.to_csv('linear_reg.gz', index=True,
#              index_label='device_id', compression="gzip")
