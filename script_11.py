# -*- coding: utf-8 -*-
"""
Created on Sun Aug 28 07:32:10 2016

@author: Tony

Changes to keras library:
After line 9, add:
import scipy.sparse as sps

After Line 827, add:
if sps.issparse(ins_batch[0]):
    ins_batch[0] = ins_batch[0].toarray()
if sps.issparse(ins_batch[1]):
    ins_batch[1] = ins_batch[1].toarray()
    
After Line 885, add:
if sps.issparse(ins_batch[0]):
    ins_batch[0] = ins_batch[0].toarray()
if sps.issparse(ins_batch[1]):
    ins_batch[1] = ins_batch[1].toarray()
    
After Line 934, add:
if sps.issparse(ins_batch[0]):
    ins_batch[0] = ins_batch[0].toarray()
if sps.issparse(ins_batch[1]):
    ins_batch[1] = ins_batch[1].toarray()
"""
#LB: 

import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import sparse
from mlxtend.classifier import StackingClassifier
from sklearn.feature_extraction import FeatureHasher, DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, scale
from sklearn.decomposition import PCA, RandomizedPCA, SparsePCA, TruncatedSVD
from sklearn.cross_validation import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.feature_selection import VarianceThreshold, SelectPercentile, f_classif, chi2
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.metrics import log_loss
from sklearn.preprocessing import MaxAbsScaler

def indicator(y):
    ind = np.zeros((y.shape[0], 12))
    ind[np.arange(y.shape[0]), y] = 1
    return ind

###########################################################
# Concatenate labels and categories to attach to each app #
###########################################################
print "Connecting labels and categories to apps"
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
print "Merging features onto apps in events/devices"
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
print "Merging features together"
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

del events

brands_models = pbd[['device_id','phone_brand', 'device_model']]
pbd.drop(['phone_brand', 'device_model'], axis=1, inplace=True)

models = LabelEncoder().fit_transform(brands_models.device_model)
brands = LabelEncoder().fit_transform(brands_models.phone_brand)
brands_models.device_model = pd.Series(models, index=brands_models.index)
brands_models.phone_brand = pd.Series(brands, index=brands_models.index)

pbd_noev = pbd[pbd.app_id == ""]
pbd_ev = pbd[pbd.app_id != ""]
del pbd
pbd_noev.drop(['app_id', 'categories', 'labels'], axis=1, inplace=True)

##################
#  Train and Test
##################
print "Encoding features and combining them to sparse matrices"
train = pd.read_csv("../input/gender_age_train.csv", dtype={'device_id': np.str})
train_ev = pd.merge(train, pbd_ev, how='inner', on='device_id')
train_noev = pd.merge(train, pbd_noev, how='inner', on='device_id')
del train

y_ev = train_ev["group"]
y_noev = train_noev["group"]

le = LabelEncoder()
y_noev = le.fit_transform(y_noev)
y_ev = le.transform(y_ev)

train_ev.drop(['age', 'gender', 'group'], axis=1, inplace=True)
train_noev.drop(['age', 'gender', 'group'], axis=1, inplace=True)
train_len_ev = len(train_ev)
train_len_noev = len(train_noev)

test = pd.read_csv("../input/gender_age_test.csv", dtype={'device_id': np.str})
test_ev = pd.merge(test, pbd_ev, how='inner', on='device_id')
test_noev = pd.merge(test, pbd_noev, how='inner', on='device_id')
del test
test_dev = pd.concat((test_ev["device_id"], test_noev["device_id"]), axis=0, ignore_index=True)

del pbd_ev
del pbd_noev

# Concat
df_ev = pd.concat((train_ev, test_ev), axis=0, ignore_index=True)
df_noev = pd.concat((train_noev, test_noev), axis=0, ignore_index=True)

# Create brand and model Series in order to label encode them
brands_models_ev = pd.merge(df_ev, brands_models, how='left', on='device_id')
brands_models_noev = pd.merge(df_noev, brands_models, how='left', on='device_id')
brands_ev = brands_models_ev['phone_brand']
brands_noev = brands_models_noev['phone_brand']
models_ev = brands_models_ev['device_model']
models_noev = brands_models_noev['device_model']
del brands_models_ev
del brands_models_noev
del brands_models

dev_app = df_ev['app_id']
dev_cat = df_ev['categories']
dev_lab = df_ev['labels']

# For each device, count app_id's, labels, and category names
print "Building input matrices for devices without events"
brands_noev = sparse.csr_matrix((np.ones(len(brands_noev), dtype='int'),
                                 (np.arange(len(brands_noev)),
                                  brands_noev.as_matrix())))
models_noev = sparse.csr_matrix((np.ones(len(models_noev), dtype='int'),
                                 (np.arange(len(models_noev)),
                                  models_noev.as_matrix())))

X_sp_noev = sparse.hstack([models_noev, brands_noev], format='csr')

print "Building input matrices for devices with events"
app_vectorizer_count = CountVectorizer()
dev_app_count = app_vectorizer_count.fit_transform(dev_app)
app_vectorizer = CountVectorizer(binary=True)
dev_app = app_vectorizer.fit_transform(dev_app)

lab_vectorizer_count = CountVectorizer()
dev_lab_count = lab_vectorizer_count.fit_transform(dev_lab)
lab_vectorizer = CountVectorizer(binary=True)
dev_lab = lab_vectorizer.fit_transform(dev_lab)

cat_vectorizer_count = CountVectorizer()
dev_cat_count = cat_vectorizer_count.fit_transform(dev_cat)
cat_vectorizer = CountVectorizer(binary=True)
dev_cat = cat_vectorizer.fit_transform(dev_cat)

brands_ev = sparse.csr_matrix((np.ones(len(brands_ev), dtype='int'),
                                 (np.arange(len(brands_ev)),
                                  brands_ev.as_matrix())))
models_ev = sparse.csr_matrix((np.ones(len(models_ev), dtype='int'),
                                 (np.arange(len(models_ev)),
                                  models_ev.as_matrix())))

X_sp_ev = sparse.hstack([models_ev, dev_app, dev_lab, brands_ev], format='csr')
X_sp_ev_count = sparse.hstack([models_ev, dev_app_count, dev_lab_count, brands_ev], format='csr')

del models_ev
del brands_ev
del dev_lab
del dev_cat
del dev_app

# Split X into train/cv/test
X_train_cv_ev = X_sp_ev[:train_len_ev,:]
X_test_ev = X_sp_ev[train_len_ev:,:]

X_train_ev, X_cv_ev, y_train_ev, y_cv_ev = train_test_split(
    X_train_cv_ev, y_ev, train_size=.90, random_state=10)
    
X_train_cv_noev = X_sp_noev[:train_len_noev,:]
X_test_noev = X_sp_noev[train_len_noev:,:]

X_train_noev, X_cv_noev, y_train_noev, y_cv_noev = train_test_split(
    X_train_cv_noev, y_noev, train_size=.90, random_state=10)

#sss = StratifiedShuffleSplit(y, train_size=.9, random_state=10)
#for train_index, cv_index in sss:
#    X_train, X_cv = X_train_cv[train_index,:], X_train_cv[cv_index,:]
#    y_train, y_cv = y[train_index], y[cv_index]

##################
#  Build Models
##################

def predictor_ev():
    print "Building Neural Net classifiers for devices with events"
    n_input = X_train_ev.shape[1]
    n_train = X_train_ev.shape[0]
    
    from keras.models import Sequential
    from keras.layers import Dense, Activation
    from keras.layers.core import Dropout
    from keras.layers.advanced_activations import PReLU
    from keras.regularizers import l2
    from keras.optimizers import Adadelta
    from keras.optimizers import SGD
    from keras.wrappers.scikit_learn import KerasClassifier
    from keras.callbacks import ModelCheckpoint
    
    def create_model(n_hidden_layers=1, nodes=[50], reg=1.0, dropouts=[.5], acts=['relu']):
        n_in = n_input    
        model = Sequential()
        for i in xrange(n_hidden_layers):
            n_out = nodes[i]
            dropout = dropouts[i]
            act = acts[i]
            model.add(Dense(output_dim=n_out, input_dim=n_in, W_regularizer=l2(reg)))
            model.add(Activation(act))
            model.add(Dropout(dropout))
            n_in = n_out
        model.add(Dense(output_dim=12, W_regularizer=l2(reg)))
        model.add(Activation("softmax"))
        # Compile model
        adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])
        return model
    
    class KerasClassifier2(KerasClassifier):
            
        def __init__(self, build_fn, fn_args, random_state=0, nb_epoch=10, batch_size=500, verbose=2):
            self.random_state = random_state
            self.nb_epoch = nb_epoch
            self.batch_size = batch_size
            self.verbose = verbose
            super(KerasClassifier2, self).__init__(build_fn, **fn_args)
            self.classes_= np.arange(12)
            self.n_classes_ = 12
            self.model = build_fn(**fn_args)
            
        def fit(self, X, y, sample_weight=None):
            return super(KerasClassifier2, self).fit(X, indicator(y),
                             verbose = self.verbose, sample_weight=sample_weight,
                             validation_data=(X_cv_ev, indicator(y_cv_ev)),
                             nb_epoch=self.nb_epoch, batch_size=self.batch_size)
    
    
        def predict_proba(self, X):
            return super(KerasClassifier2, self).predict_proba(X, batch_size=500, verbose=0)
            
        def predict(self, X):
            return super(KerasClassifier2, self).predict_proba(X, batch_size=500, verbose=0)            
    
    nn1_args = {'n_hidden_layers': 2, 'nodes': [600, 400], 'reg': 1.8,
                'dropouts': [.3, .4], 'acts': ['relu', 'relu']}
    nn2_args = {'n_hidden_layers': 3, 'nodes': [300, 100, 50], 'reg': 2.0,
                'dropouts': [.2, .4, .5], 'acts': ['relu', 'relu', 'relu']}
    nn3_args = {'n_hidden_layers': 4, 'nodes': [1001, 511, 245, 99], 'reg': 2.0,
                'dropouts': [.2, .3, .2, .3], 'acts': ['relu', 'relu', 'relu', 'relu']}
    nn4_args = {'n_hidden_layers': 1, 'nodes': [500], 'reg': 1.2,
                'dropouts': [.25], 'acts': ['relu']}
    nn5_args = {'n_hidden_layers': 5, 'nodes': [1343, 1012, 757, 539, 117],
                'reg': 2.5, 'dropouts': [.2, .3, .4, .4, .4],
                'acts': ['relu', 'relu', 'relu', 'relu', 'relu']}
    
    clfNN1 = KerasClassifier2(create_model, nn1_args, random_state=5, nb_epoch=5)
    clfNN2 = KerasClassifier2(create_model, nn2_args, random_state=23, nb_epoch=11)
    clfNN3 = KerasClassifier2(create_model, nn3_args, random_state=710, nb_epoch=6)
    clfNN4 = KerasClassifier2(create_model, nn4_args, random_state=5072, nb_epoch=6)
    clfNN5 = KerasClassifier2(create_model, nn5_args, random_state=2016, nb_epoch=12)
    
    print "Building XGBoost classifiers for devices with events"
    xgb_params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster": "gblinear",
    "max_depth": 6,
    "eval_metric": "mlogloss",
    "eta": 0.07,
    "silent": 1,
    "alpha": 3.5,
    }
    
    class XGBClassifier2(xgb.XGBClassifier):
    
        def __init__(self, max_depth=xgb_params['max_depth'],
                     objective='multi:softprob', missing=None, 
                     learning_rate=xgb_params['eta'], n_estimators=40, subsample=1,
                     reg_alpha=xgb_params['alpha'], seed=2016, booster='gblinear'):
            super(XGBClassifier2, self).__init__(max_depth=max_depth, seed=seed,
                        objective=objective, missing=missing,
                        learning_rate=learning_rate, n_estimators=n_estimators,
                        subsample=subsample, reg_alpha=reg_alpha)
            self.booster = xgb_params['booster']
            
        def fit(self, X, y):
            super(XGBClassifier2, self).fit(X.tocsc(), y, eval_metric='mlogloss',
                                            eval_set=[(X_cv_ev.tocsc(), y_cv_ev)])
    
    gbm1 = XGBClassifier2(seed=0, booster='gblinear', n_estimators=28)
    gbm2 = XGBClassifier2(seed=6, booster='gblinear', n_estimators=28)
    gbm3 = XGBClassifier2(seed=151, booster='gbtree', n_estimators=28)
    gbm4 = XGBClassifier2(seed=1047, booster='gbtree', n_estimators=28)
    gbm5 = XGBClassifier2(seed=22, booster='dart', n_estimators=28)
    
    print "Building Logistic Regression classifier for devices with events"
    clfLR = LogisticRegression(C=.02, random_state=2016, multi_class='multinomial', solver='newton-cg')
    
    #Combine results of classifiers
    print "Stacking classifiers for devices with events"
    clf_ls = [gbm1,gbm2,gbm3,gbm4,gbm5,clfNN1,clfNN2,clfNN3,clfNN4,clfNN5,clfLR]
    meta = LogisticRegression()
    stack = StackingClassifier(clf_ls, meta, use_probas=True, verbose=1)
    
    stack.fit(X_train_ev, y_train_ev)
    print log_loss(y_cv_ev, stack.predict_proba(X_cv_ev))
    y_pred_ev = stack.predict_proba(X_test_ev)
    #y_pre = (pred_prob_nn+y_pre)/2.0
    return y_pred_ev
    
#def predictor_noev():
print "Building Neural Net classifiers for devices with no events"
n_input = X_train_noev.shape[1]
n_train = X_train_noev.shape[0]

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.core import Dropout
from keras.layers.advanced_activations import PReLU
from keras.regularizers import l2
from keras.optimizers import Adadelta
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ModelCheckpoint

def create_model(n_hidden_layers=1, nodes=[50], reg=1.0, dropouts=[.5], acts=['relu']):
    n_in = n_input    
    model = Sequential()
    for i in xrange(n_hidden_layers):
        n_out = nodes[i]
        dropout = dropouts[i]
        act = acts[i]
        model.add(Dense(output_dim=n_out, input_dim=n_in, W_regularizer=l2(reg)))
        model.add(Activation(act))
        model.add(Dropout(dropout))
        n_in = n_out
    model.add(Dense(output_dim=12, W_regularizer=l2(reg)))
    model.add(Activation("softmax"))
    # Compile model
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08)
    sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=adadelta, metrics=['accuracy'])
    return model

class KerasClassifier2(KerasClassifier):
        
    def __init__(self, build_fn, fn_args, random_state=0, nb_epoch=10, batch_size=500, verbose=2):
        self.random_state = random_state
        self.nb_epoch = nb_epoch
        self.batch_size = batch_size
        self.verbose = verbose
        super(KerasClassifier2, self).__init__(build_fn, **fn_args)
        self.classes_= np.arange(12)
        self.n_classes_ = 12
        self.model = build_fn(**fn_args)
        
    def fit(self, X, y, sample_weight=None):
        return super(KerasClassifier2, self).fit(X, indicator(y),
                         verbose = self.verbose, sample_weight=sample_weight,
                         validation_data=(X_cv_noev, indicator(y_cv_noev)),
                         nb_epoch=self.nb_epoch, batch_size=self.batch_size)


    def predict_proba(self, X):
        return super(KerasClassifier2, self).predict_proba(X, batch_size=500, verbose=0)
        
    def predict(self, X):
        return super(KerasClassifier2, self).predict_proba(X, batch_size=500, verbose=0)            

nn1_args = {'n_hidden_layers': 2, 'nodes': [600, 400], 'reg': 1.5,
            'dropouts': [.3, .4], 'acts': ['relu', 'relu']}
nn2_args = {'n_hidden_layers': 3, 'nodes': [300, 100, 50], 'reg': 2.0,
            'dropouts': [.2, .4, .5], 'acts': ['relu', 'relu', 'relu']}
nn3_args = {'n_hidden_layers': 4, 'nodes': [1001, 511, 245, 99], 'reg': 2.0,
            'dropouts': [.2, .3, .2, .3], 'acts': ['relu', 'relu', 'relu', 'relu']}
nn4_args = {'n_hidden_layers': 1, 'nodes': [500], 'reg': 1.2,
            'dropouts': [.25], 'acts': ['relu']}
nn5_args = {'n_hidden_layers': 5, 'nodes': [1343, 1012, 757, 539, 117],
            'reg': 2.5, 'dropouts': [.2, .3, .4, .4, .4],
            'acts': ['relu', 'relu', 'relu', 'relu', 'relu']}

clfNN1 = KerasClassifier2(create_model, nn1_args, random_state=5, nb_epoch=6)
clfNN2 = KerasClassifier2(create_model, nn2_args, random_state=23, nb_epoch=9)
clfNN3 = KerasClassifier2(create_model, nn3_args, random_state=710, nb_epoch=10)
clfNN4 = KerasClassifier2(create_model, nn4_args, random_state=5072, nb_epoch=4)
clfNN5 = KerasClassifier2(create_model, nn5_args, random_state=2016, nb_epoch=12)

print "Building XGBoost classifiers for devices with no events"
xgb_params = {
"objective": "multi:softprob",
"num_class": 12,
"booster": "gblinear",
"max_depth": 6,
"eval_metric": "mlogloss",
"eta": 0.07,
"silent": 1,
"alpha": 3.5,
}

class XGBClassifier2(xgb.XGBClassifier):

    def __init__(self, max_depth=xgb_params['max_depth'],
                 objective='multi:softprob', missing=None, 
                 learning_rate=xgb_params['eta'], n_estimators=40, subsample=1,
                 reg_alpha=xgb_params['alpha'], seed=2016, booster='gblinear'):
        super(XGBClassifier2, self).__init__(max_depth=max_depth, seed=seed,
                    objective=objective, missing=missing,
                    learning_rate=learning_rate, n_estimators=n_estimators,
                    subsample=subsample, reg_alpha=reg_alpha)
        self.booster = xgb_params['booster']
        
    def fit(self, X, y):
        super(XGBClassifier2, self).fit(X.tocsc(), y, eval_metric='mlogloss',
                                        eval_set=[(X_cv_noev.tocsc(), y_cv_noev)])

gbm1 = XGBClassifier2(seed=0, booster='gblinear', n_estimators=28)
gbm2 = XGBClassifier2(seed=6, booster='gblinear', n_estimators=28)
gbm3 = XGBClassifier2(seed=151, booster='gbtree', n_estimators=28)
gbm4 = XGBClassifier2(seed=1047, booster='gbtree', n_estimators=28)
gbm5 = XGBClassifier2(seed=22, booster='dart', n_estimators=28)

print "Building logistic regression classifier for devices with no events"
clfLR = LogisticRegression(C=.02, random_state=2016, multi_class='multinomial', solver='newton-cg')

#KNN
#clfKNN = KNeighborsClassifier(n_neighbors=5)
#clfKNN.fit(X_train_noev, y_train_noev)
#print log_loss(y_cv_noev, clfKNN.predict_proba(X_cv_noev))
#
##NB
#clfNB = MultinomialNB(alpha=1.0)
#clfNB.fit(X_train_noev, y_train_noev)
#print log_loss(y_cv_noev, clfNB.predict_proba(X_cv_noev))

#Combine results of classifiers
print "Stacking classifiers for devices with no events"
clf_ls = [gbm1,gbm2,gbm3,gbm4,gbm5,clfNN1,clfNN2,clfNN3,clfNN4,clfNN5,clfLR]
meta = LogisticRegression()
stack = StackingClassifier(clf_ls, meta, use_probas=True, verbose=1)

stack.fit(X_train_noev, y_train_noev)
print log_loss(y_cv_noev, stack.predict_proba(X_cv_noev))
y_pred_noev = stack.predict_proba(X_test_noev)
    #y_pre = (pred_prob_nn+y_pre)/2.0
#    return y_pred_noev

y_pred_ev = predictor_ev()
#y_pred_noev = predictor_noev()

# Write results
result = pd.DataFrame(np.hstack(y_pred_ev, y_pred_noev), columns=le.classes_)
result["device_id"] = test_dev
result = result.set_index("device_id")
result.to_csv('stacking_1.gz', index=True,
              index_label='device_id', compression="gzip")
