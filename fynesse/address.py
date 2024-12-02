# This file contains code for suporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""

"""Address a particular question that arises from the data"""
import random
import pandas as pd
import statsmodels.api as sm
from scipy.stats import pearsonr


def k_fold(k, X, y, alpha=0, L1_wt=0):
  indexes =[i for i in range(len(X))]
  random.shuffle(indexes)
  X  = X.iloc[indexes]
  y  = y.iloc[indexes]
  start = 0
  end = 0
  predicted = []


  remainder = len(indexes)%k
  length = len(indexes)//k

  for i in range (0,k):
    end = end + length
    if remainder > 0:
      end = end + 1
      remainder = remainder - 1

    testing_design = X.iloc[start:end]
    testing_y = y.iloc[start:end]
    training_design = pd.concat([X.iloc[0:start], X.iloc[end:len(indexes)]])
    training_y = pd.concat([y.iloc[0:start], y.iloc[end:len(indexes)]])

    # Fit a simple linear model
    m_linear_basis = sm.OLS(training_y, training_design)
    results_basis = m_linear_basis.fit_regularized(alpha = alpha, L1_wt = L1_wt)

    #uses a design matrix of non-linear functions.
    #y_pred_linear_basis = results_basis.get_prediction(testing_design).summary_frame()
    y_pred_linear_basis = results_basis.predict(testing_design)
    
    predicted = predicted + list(y_pred_linear_basis)
    start = end

  correlation, p_value = pearsonr(y, predicted)
  return(correlation, p_value)
