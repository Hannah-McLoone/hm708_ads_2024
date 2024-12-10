from .config import *
import requests
from . import access

"""These are the types of import we might expect in this file
import pandas
import bokeh
import seaborn
import matplotlib.pyplot as plt
import sklearn.decomposition as decomposition
import sklearn.feature_extraction"""

"""Place commands in this file to assess the data you have downloaded. How are missing values encoded, how are outliers encoded? What do columns represent, makes rure they are correctly labeled. How is the data indexed. Crete visualisation routines to assess the data (e.g. in bokeh). Ensure that date formats are correct and correctly timezoned."""

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import requests  # Added import for requests
from matplotlib.colors import Normalize  # Added import for Normalize


def data():
    """Load the data from access and ensure missing values are correctly encoded as well as indices correct, column names informative, date and times correctly formatted. Return a structured data structure such as a data frame."""
    df = access.data()
    raise NotImplementedError

def query(data):
    """Request user input for some aspect of the data."""
    raise NotImplementedError

def view(data):
    """Provide a view of the data that allows the user to verify some aspect of its quality."""
    raise NotImplementedError

def labelled(data):
    """Provide a labelled set of data ready for supervised learning."""
    raise NotImplementedError


def querying_count_for_eng_wls(criteria):
  total = 0
  overpass_url = "https://overpass-api.de/api/interpreter"

  for country in ['ENG','WLS']:
    query = f"""
    [out:json][timeout:50];
    area["ISO3166-2"="GB-{country}"]->.searchArea;
    node[{criteria}](area.searchArea);
    out count;
    """
    response = requests.post(overpass_url, data={'data': query})

    if response.status_code == 200:
        data = response.json()
        total = total + int(data.get('elements', [{}])[0].get('tags', {}).get('total', 'Unknown'))

  return total




def elbow(df, max_k = 20):
  scaler = StandardScaler()
  df = scaler.fit_transform(df)
  inertia = []
  for k in range(1, max_k):
      kmeans = KMeans(n_clusters=k, random_state=0)
      kmeans.fit(df)
      inertia.append(kmeans.inertia_)

  plt.plot(range(1, max_k), inertia, marker='o')
  plt.xlabel('Number of clusters')
  plt.ylabel('Inertia')
  plt.title('Elbow Method')
  plt.show()






def convert_to_percentage_increase(df):
  # Convert all columns to numeric, replacing non-numeric values with NaN
  df = df.apply(pd.to_numeric, errors='coerce')

  # Replace zeros with NaN to treat them as missing values for interpolation
  df = df.replace(0, np.nan)

  # Interpolate all columns (vectorized operation)
  df = df.interpolate(axis=1, limit_direction='both')

  # Compute percentage increase along rows (vectorized operation)
  df_percentage_increase = df.pct_change(axis=1) * 100

  # Fill NaN with 0
  df_percentage_increase = df_percentage_increase.fillna(0)
  return df_percentage_increase





def plot_by_cluster(df, y_range = (-100,5000000)):
  data = df.drop(columns='cluster')  # Drop the 'cluster' column to use the time series data
  clusters = df['cluster']

  colors = plt.cm.get_cmap('tab10', len(clusters.unique()))

  plt.figure(figsize=(10, 6))  # Adjust the figure size if necessary!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

  for i, cluster_id in enumerate(clusters.unique()):
      cluster_data = data[clusters == cluster_id]
      for _, row in cluster_data.iterrows():
          plt.plot(row.index, row.values, color=colors(i), alpha=0.7)

  # Title and labels
  plt.title('Time Series Data Colored by Cluster')
  plt.xlabel('Time')
  plt.ylabel('percentage increase')
  plt.ylim(y_range[0], y_range[1])
  plt.show()


def silhouette(df):
  scaler = StandardScaler()
  df = scaler.fit_transform(df)
  silhouette_scores = []
  k_range = range(2, 20)

  for k in k_range:
      kmeans = KMeans(n_clusters=k, random_state=0)
      labels = kmeans.fit_predict(df)
      score = silhouette_score(df, labels)
      silhouette_scores.append(score)

  # plot
  plt.plot(k_range, silhouette_scores, marker='o')
  plt.xlabel('Number of clusters')
  plt.ylabel('Silhouette Score')
  plt.title('Silhouette Method')
  plt.show()


def plot_sample_of_cluster(df,number = 100, y_range = (-100,5000000)):
  data = df.drop(columns='cluster')
  clusters = df['cluster']

  # Get a colormap with a distinct color for each cluster
  colors = plt.cm.get_cmap('tab10', len(clusters.unique()))

  # Set up the figure
  plt.figure(figsize=(10, 6))

  # Loop through each unique cluster
  for i, cluster_id in enumerate(clusters.unique()):
      # Filter data for the current cluster
      cluster_data = data[clusters == cluster_id]

      # Sample 100 random rows from each cluster (if there are at least 100)
      sampled_data = cluster_data.sample(n=number, random_state=42) if len(cluster_data) >= 100 else cluster_data

      # Plot each of the sampled rows
      for _, row in sampled_data.iterrows():
          plt.plot(row.index, row.values, color=colors(i), alpha=0.3)

  # Show the plot
  plt.title('Time Series Data Colored by Cluster')
  plt.xlabel('Time')
  plt.ylim(y_range[0], y_range[1])
  plt.ylabel('percentage increase')
  plt.show()




def map_census_value_by_colour(conn,table,column,scale):
  cur = conn.cursor()
  cur.execute(f"SELECT geography_code, {column} FROM {table};")
  census = cur.fetchall()
  census = pd.DataFrame(census)
  census.columns = ['geography_code', column]


  town_coords = pd.read_csv('Middle_layer_Super_Output_Areas_December_2021_Boundaries_EW_BSC_V3_-5990297399386808643.csv')
  joined = census.merge(town_coords,  left_on=['geography_code'], right_on = ['MSOA21CD'], how='inner')

  # Define a custom normalization focusing on the 0.025 to 0.125 range
  norm = Normalize(vmin = scale[0], vmax=scale[1])
  sizes = joined['Shape__Area'] /2000000


  plt.figure(figsize=(10, 6))
  scatter = plt.scatter(
      joined['LONG'], joined['LAT'],
      c=joined[column],
      cmap='viridis',
      norm=norm,  # Custom normalization
      s=sizes,    # Set the size of the dots proportional to 'Shape__Area'
  )

  plt.colorbar(scatter, label='Proportion')
  plt.title(table + " : "+column)
  plt.show()
