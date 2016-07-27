# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 08:14:34 2016

@author: Tony
"""

#### Libraries
# Standard library
import cPickle
import gzip

# Third-party libraries
import numpy as np
import theano
import theano.tensor as T
from theano.tensor.nnet import softmax
from theano.tensor import shared_randomstreams

# Activation functions for neurons (rectified linear units = relu)
from theano.tensor.nnet import relu
from theano.tensor.nnet import sigmoid
from theano.tensor import tanh

# GPU on or off
GPU = False
if GPU:
    print "Using GPU."
    try: theano.config.device = 'gpu'
    except: pass
    theano.config.floatX = 'float32'
else:
    print "Using CPU"
    
#### Helper functions
def dropout_layer(layer, p_dropout):
    srng = shared_randomstreams.RandomStreams(seed=9999)
    mask = srng.binomial(n=1, p=1-p_dropout, size=layer.shape)
    return layer * T.cast(mask, theano.config.floatX)

#### Load data for GPU
def load_data_into_shared(X_train, y_train, X_cv, y_cv, X_test = False):
    # Put data into shared vars for GPU usage
    X_train_shared = theano.shared(np.asarray(X_train, dtype=theano.config.floatX), borrow=True)
    y_train_shared = theano.shared(np.asarray(y_train, dtype='int32'), borrow=True)
    X_cv_shared = theano.shared(np.asarray(X_cv, dtype=theano.config.floatX), borrow=True)
    y_cv_shared = theano.shared(np.asarray(y_cv, dtype='int32'), borrow=True)
    if type(X_test) == bool:
        X_test_shared = X_test
    else:
        X_test_shared = theano.shared(np.asarray(X_test, dtype=theano.config.floatX), borrow=True)
    return X_train_shared, y_train_shared, X_cv_shared, y_cv_shared, X_test_shared
    
class Network(object):
    def __init__(self, layers, num_batch, epochs=10, eta=1.0, lmb=0.0):
        '''
        Takes in a list of layers for the architecture as well as the num of
        pieces of data for each iteration of mini-batch gradient descent
        while training the network
        '''
        self.layers = layers
        self.num_batch = num_batch
        self.epochs = epochs
        self.eta = eta
        self.lmb = lmb
        self.params = [params for layer in self.layers for params in layer.params]
        self.X = T.matrix('X')
        self.y = T.ivector('y')
        act = self.X
        for layer in self.layers:
            layer.setup_layer(act, num_batch)
            act = layer.outputs
        self.outputs = act
    
    def fit(self, X_train, y_train, X_cv, y_cv):
        '''
        Use mini-batch gradient descent to optimize weight and bias matrices.
        May use faster method after initial implementation.
        '''        
        
        num_batch_train = X_train.get_value(borrow=True).shape[0] / self.num_batch
        num_batch_cv = X_cv.get_value(borrow=True).shape[0] / self.num_batch
        
        L2_reg = sum([(layer.w ** 2).sum() for layer in self.layers])
        J = self.layers[-1].cost(self) + self.lmb * L2_reg / (num_batch_train * 2.0)
        grads = T.grad(J, self.params)
        updates = [(param, param-self.eta*grad) for param, grad in zip(self.params, grads)]
        
        i = T.iscalar()
        batch_train = theano.function([i], J, updates=updates, givens={
                self.X: X_train[i*self.num_batch:(i+1)*self.num_batch],
                self.y: y_train[i*self.num_batch:(i+1)*self.num_batch]})
        
        cost_train = theano.function([i], self.layers[-1].cost(self), givens={
                self.X: X_train[i*self.num_batch: (i+1)*self.num_batch],
                self.y: y_train[i*self.num_batch: (i+1)*self.num_batch]})
        
        cost_cv = theano.function([i], self.layers[-1].cost(self), givens={
                self.X: X_cv[i*self.num_batch: (i+1)*self.num_batch], 
                self.y: y_cv[i*self.num_batch: (i+1)*self.num_batch]})
        
        self.cv_predictions = theano.function([i], self.layers[-1].y_out, givens={
                self.X: X_cv[i*self.num_batch: (i+1)*self.num_batch]})
        
        best_cv_cost = np.Infinity
        cost_ij = np.ones((self.epochs, num_batch_train)) * -1
        iteration = 0
        best_epoch = -1
        for epoch_i in xrange(self.epochs):
#            print("Training epoch number {0}".format(epoch_i))
            for batch_j in xrange(num_batch_train):                  
                cost_ij[epoch_i, batch_j] = batch_train(batch_j)
            iteration += num_batch_train
            cost_cv_tracker = np.mean([cost_cv(j) for j in xrange(num_batch_cv)])
            print("epoch {0}: cv cost {1}".format(epoch_i, cost_cv_tracker))
            if cost_cv_tracker < best_cv_cost:
#                print("This is the best cv cost to date.")
                best_cv_cost = cost_cv_tracker
                best_epoch = epoch_i
                corresponding_train_cost = np.mean([cost_train(j) for j in xrange(num_batch_train)])
                trained_params = list(self.params)
        print "Completed training."
        print "Best cv cost of {0} obtained at epoch {1}".format(best_cv_cost, best_epoch)
        print "Corresponding train cost of {0}".format(corresponding_train_cost)
        self.params = trained_params
        
    def predict_proba(self, X_test):
        inputs = X_test
        for layer in self.layers[:-1]:
            inputs = layer.a_fn(T.dot(inputs, layer.w) + layer.b)
        return softmax(T.dot(inputs, self.layers[-1].w) + self.layers[-1].b).eval()
        
class HiddenLayer(object):
    rand_seed = 0
    def __init__(self, num_inputs, num_outputs, a_fn = relu, p_dropout=0.0):
        self.num_inputs = num_inputs
        self.num_outouts = num_outputs
        self.a_fn = a_fn
        self.p_dropout = p_dropout
        rng = np.random.RandomState(self.rand_seed)
        HiddenLayer.rand_seed += 1
        self.w = theano.shared(rng.normal(scale=np.sqrt(1.0 / num_outputs), \
            size=(num_inputs, num_outputs)), name='w', borrow=True)
        self.b = theano.shared(rng.normal(size=(num_outputs,)), name='b', borrow=True)
        self.params = [self.w, self.b]
        
    def setup_layer(self, inputs, num_batch):
        self.inputs = dropout_layer(inputs.reshape((num_batch, self.num_inputs)), self.p_dropout)
        self.outputs = self.a_fn(T.dot(self.inputs, self.w) + self.b)
        self.y_out = T.argmax(self.outputs, axis=1)

        
class SoftmaxLayer(object):
    rand_seed = 5000
    def __init__(self, num_inputs, num_outputs, p_dropout=0.0):
        self.num_inputs = num_inputs
        self.num_outouts = num_outputs
        self.p_dropout = p_dropout
        rng = np.random.RandomState(self.rand_seed)
        SoftmaxLayer.rand_seed += 1
        self.w = theano.shared(rng.normal(scale=np.sqrt(1.0 / num_outputs), \
            size=(num_inputs, num_outputs)), name='w', borrow=True)
        self.b = theano.shared(rng.normal(size=(num_outputs,)), name='b', borrow=True)
        self.params = [self.w, self.b]
        
    def setup_layer(self, inputs, num_batch):
        self.inputs = dropout_layer(inputs.reshape((num_batch, self.num_inputs)), self.p_dropout)
        self.outputs = softmax(T.dot(self.inputs, self.w) + self.b)
        self.y_out = T.argmax(self.outputs, axis=1)
        
    def cost(self, net):
        #return T.nnet.categorical_crossentropy(self.outputs, net.y)
        return -T.mean(T.log(self.outputs)[T.arange(net.y.shape[0]), net.y])
    
    def accuracy(self, y):
        return T.mean(T.eq(y, self.y_out))
