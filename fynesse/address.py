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
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import numpy as np
import requests
from sklearn.preprocessing import StandardScaler





def k_fold(k, X, y, alpha=0, L1_wt=0, plot = False):
  indexes =[i for i in range(len(X))]
  X = X.astype(float)
  y = y.astype(float)
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

    if plot:
      plt.scatter(testing_y, y_pred_linear_basis)


  if plot:
    plt.xlim(plot[0][0], plot[0][1])
    plt.ylim(plot[1][0], plot[1][1])
    plt.show()

  if isinstance(y, pd.DataFrame):
    y = y.values.flatten().tolist()
  correlation, p_value = pearsonr(y, predicted)
  return(correlation, p_value)




def pca_k_fold(k, X, y, pca_components, alpha=0, L1_wt=0, plot = False):
  scaler = StandardScaler()
  pca = PCA(n_components=pca_components)


  indexes =[i for i in range(len(X))]
  X = X.astype(float)
  y = y.astype(float)
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



    #-----pca
# Scale the data
    training_design = scaler.fit_transform(training_design)
    testing_design = scaler.transform(testing_design)
    
    # PCA transformation
    pca.fit(training_design)
    training_design = pca.transform(training_design)
    testing_design = pca.transform(testing_design)
    
    # Add constant for intercept
    training_design = sm.add_constant(training_design, has_constant='add')
    testing_design = sm.add_constant(testing_design, has_constant='add')
    
    # Fit OLS model
    m_linear_basis = sm.OLS(training_y, training_design).fit()
    m_linear_basis = sm.OLS(training_y, training_design)
    results_basis = m_linear_basis.fit_regularized(alpha = alpha, L1_wt = L1_wt)
    y_pred_linear_basis = results_basis.predict(testing_design)



    predicted = predicted + list(y_pred_linear_basis)
    start = end

    if plot:
      plt.scatter(testing_y, y_pred_linear_basis)


  if plot:
    plt.xlim(plot[0][0], plot[0][1])
    plt.ylim(plot[1][0], plot[1][1])
    plt.show()

  if isinstance(y, pd.DataFrame):
    y = y.values.flatten().tolist()
  correlation, p_value = pearsonr(y, predicted)
  return(correlation, p_value)



def find_nearest_feature(lat, lon,condition):
    # Define the Overpass API URL
    overpass_url = "http://overpass-api.de/api/interpreter"

    # Overpass QL query to find the nearest university
    query = f"""
    [out:json];
    node
      [{condition}]
      (around:50000,{lat},{lon});  // Searches within 50 km radius
    out body;
    """

    try:
        # Send the request to the Overpass API
        response = requests.get(overpass_url, params={'data': query})
        response.raise_for_status()  # Raise HTTPError for bad responses

        # Parse the JSON response
        data = response.json()
        if 'elements' in data and len(data['elements']) > 0:
            # Sort results by distance (in case of multiple results)
            universities = sorted(
                data['elements'],
                key=lambda x: ((x['lat'] - lat) ** 2 + (x['lon'] - lon) ** 2) ** 0.5
            )

            # Return the nearest university
            nearest = universities[0]
            return ((nearest['lat'] - lat) ** 2 + (nearest['lon'] - lon) ** 2) ** 0.5
        else:
            return "No universities found within the search radius."

    except requests.exceptions.RequestException as e:
        return f"An error occurred: {e}"
