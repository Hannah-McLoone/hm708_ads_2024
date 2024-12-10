from .config import *
import requests
import pymysql
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import geopandas as gpd
import numpy as np
import osmnx as ox
import pymysql
import re
import zipfile
import io
import os
import sqlite3
import csv
from shapely.geometry import Point, Polygon

"""These are the types of import we might expect in this file
import httplib2
import oauth2
import tables
import mongodb
import sqlite"""

# This file accesses the data

"""Place commands in this file to access the data electronically. Don't remove any missing values, or deal with outliers. Make sure you have legalities correct, both intellectual property and personal data privacy rights. Beyond the legal side also think about the ethical issues around this data. """

def data():
    """Read the data from the web or local file, returning structured format such as a data frame"""
    raise NotImplementedError


def count_pois_near_coordinates(latitude: float, longitude: float, tags: dict, distance_km: float = 1.0) -> dict:
    """
    Count Points of Interest (POIs) near a given pair of coordinates within a specified distance.
    Args:
        latitude (float): Latitude of the location.
        longitude (float): Longitude of the location.
        tags (dict): A dictionary of OSM tags to filter the POIs (e.g., {'amenity': True, 'tourism': True}).
        distance_km (float): The distance around the location in kilometers. Default is 1 km.
    Returns:
        dict: A dictionary where keys are the OSM tags and values are the counts of POIs for each tag.
    """


    #this approximation is only accurate at the equator, as you get closer to the poles it is less accurate
    box_width = distance_km * 0.008
    box_height = distance_km * 0.008

    #when we say 'around', we mean the square box enclosing the area. not a circular surrounding area
    north = latitude + box_height/2
    south = latitude - box_width/2
    west = longitude - box_width/2
    east = longitude + box_width/2
    poi_counts = {}
    try:
      pois = ox.features_from_bbox((west, south, east, north), tags)
    except:
      for tag, tag_values in tags.items():
            if isinstance(tag_values, list):
              for value in tag_values:
                poi_counts[f"{tag}:{value}"] = 0

            else:
              poi_counts[tag] = 0
      return poi_counts

    pois_df = pd.DataFrame(pois)

    #count the occurrences of every tag
    for tag, tag_values in tags.items():
          if isinstance(tag_values, list):
            for value in tag_values:
              if tag in pois_df.columns:
                poi_counts[f"{tag}:{value}"] = max((pois_df[tag] == value).sum(),0)
              else:
                poi_counts[f"{tag}:{value}"] = 0

          else:
            if tag in pois_df.columns:
              poi_counts[tag] = max(pois_df[tag].notnull().sum(),0)
            else:
              poi_counts[tag] = 0

    return poi_counts


def download_census_data(code, base_dir=''):
  url = f'https://www.nomisweb.co.uk/output/census/2021/census2021-{code.lower()}.zip'
  extract_dir = os.path.join(base_dir, os.path.splitext(os.path.basename(url))[0])

  if os.path.exists(extract_dir) and os.listdir(extract_dir):
    print(f"Files already exist at: {extract_dir}.")
    return

  os.makedirs(extract_dir, exist_ok=True)
  response = requests.get(url)
  response.raise_for_status()

  with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.extractall(extract_dir)

  print(f"Files extracted to: {extract_dir}")



def load_census_data(code, level='msoa'):
  return pd.read_csv(f'census2021-{code.lower()}/census2021-{code.lower()}-{level}.csv')


def calculate_ratio(dataframe,population, columns = None):
  if columns == None:
    columns = dataframe.select_dtypes(include=['int', 'float']).columns

  for column in columns:
    if column != population and column != 'date':
      dataframe[column] = dataframe[column] / dataframe[population]

  return dataframe



def merge_on_town(dataframe):
  #dataframe.drop(['geography code'],axis=1,inplace=True)

  dataframe['geography'] = dataframe['geography'].str.replace(r'\d+$', '', regex=True).str.strip()
  dataframe = dataframe.groupby('geography', as_index=False).sum()
  return dataframe

def create_town_name(dataframe):
  dataframe['town_name'] = dataframe['geography'].str.replace(r'\d+$', '', regex=True).str.strip()
  return dataframe

def filter_on_date(dataframe, date):
  dataframe = dataframe[dataframe['date'] == date]
  return dataframe


def calculate_ratio_dirty(df, population, columns=None):
    #for dirty data
    df_copy = df.copy()
    if columns is None:
        columns = df_copy.columns

    for column in columns:
        if column != population and column != 'date':
            # Convert non-numeric values to NaN
            df_copy[column] = pd.to_numeric(df_copy[column], errors='coerce')
            # Perform the division, ignoring NaN values
            df_copy[column] = df_copy[column] / df_copy[population]

    return df_copy.fillna(df)




def count_decimal_digits(number):
  # Convert number to string
  str_number = str(number)

  # Check if there is a decimal point
  if '.' in str_number:
      # Split the number at the decimal point and count the digits after it
      decimal_part = str_number.split('.')[1]
      return len(decimal_part)
  else:
      # If there is no decimal point, return 0
      return 0


def uplaod_df_to_aws(conn, df,table_name):
  temp_csv_file = "temp_data.csv"
  df.to_csv(temp_csv_file, index=False, header=False)
  #sql = f"""LOAD DATA LOCAL INFILE "temp_data.csv" INTO TABLE `{table_name}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"' LINES STARTING BY '' TERMINATED BY '\n';"""
  #%sql {sql}
  #os.remove(temp_csv_file)
  cur = conn.cursor()
  cur.execute(f"""LOAD DATA LOCAL INFILE "temp_data.csv" INTO TABLE `{table_name}` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '"' LINES STARTING BY '' TERMINATED BY '\n';""")
  conn.commit()
  os.remove(temp_csv_file)


#---------------------deprecated-------------------------------------------------------


def add_feature_column(conn,condition,col_name):
  cur = conn.cursor()

  # Add the new column to the table
  query = f"ALTER TABLE `features_by_MSOA` ADD {col_name} INTEGER"
  cur.execute(query)
  conn.commit()

  # Fetch distinct location names
  cur.execute("SELECT DISTINCT `location_name` FROM `features_by_MSOA`")
  towns = [row[0] for row in cur.fetchall()]

  progress = 0

  for town in towns:
    print("done towns:",progress)
    progress = progress+1

    cur.execute(f"SELECT `latitude`, `longitude` FROM `features_by_MSOA` WHERE location_name = '{town}';")
    centres = cur.fetchall()

    # Get POI coordinates based on the condition
    pois = get_feature_coords_from_town(town, condition)
        
    centres = np.array(centres)
    pois = np.array(pois)
    centre_count = {(centre[0],centre[1]): 0 for centre in centres}
    #accounting for the areas that are labeled differently under osm. would have done this more thorougly if pursued idea futher
    if len(pois) == 0:
      london_town = 'London Borough of '+ town
      pois = get_feature_coords_from_town(london_town,condition)
      pois = np.array(pois)


    for p in pois:
      distances = np.linalg.norm(centres - p, axis=1)

      # Find the index of the smallest distance
      closest_index = np.argmin(distances)

      # Get the centre with the smallest distance
      closest_point = centres[closest_index]
      centre_count[(closest_point[0],closest_point[1])] = centre_count[(closest_point[0],closest_point[1])] + 1

    for centre in centre_count:
      lat = centre[0]
      lat_dig = count_decimal_digits(lat)
      lon = centre[1]
      lon_dig = count_decimal_digits(lon)
      number_counted = centre_count[centre]


      query = f"UPDATE `features_by_MSOA` SET {col_name} = {number_counted} WHERE ROUND(`latitude`, {lat_dig}) = {lat} AND ROUND(`longitude`, {lon_dig}) = {lon};"
      cur.execute(query)

    conn.commit()
  cur.close()


def get_feature_coords_from_town(town,feature):
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f"""
    [out:json][timeout:50];
    area[name="{town}"]->.searchArea;
    node[{feature}](area.searchArea);
    out body;
    """

    response = requests.post(overpass_url, data={'data': query})
    if response.status_code == 200:
        data = response.json()
        data = data.get('elements', [{}])
        coords = [(d['lat'],d['lon']) for d in data if 'lat' in d and 'lon' in d]
        return(coords)
    else:
        print(f"Failed to retrieve data. HTTP Status Code: {response.status_code}")
        print(f"Response: {response.text}")


#-------------------previous exercises--------------------------------------------------



def download_price_paid_data(year_from, year_to):
    # Base URL where the dataset is stored 
    base_url = "http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com"
    """Download UK house price data for given year range"""
    # File name with placeholders
    file_name = "/pp-<year>-part<part>.csv"
    for year in range(year_from, (year_to+1)):
        print (f"Downloading data for year: {year}")
        for part in range(1,3):
            url = base_url + file_name.replace("<year>", str(year)).replace("<part>", str(part))
            response = requests.get(url)
            if response.status_code == 200:
                with open("." + file_name.replace("<year>", str(year)).replace("<part>", str(part)), "wb") as file:
                    file.write(response.content)


def create_connection(user, password, host, database, port=3306):
    """ Create a database connection to the MariaDB database
        specified by the host url and database name.
    :param user: username
    :param password: password
    :param host: host url
    :param database: database name
    :param port: port number
    :return: Connection object or None
    """
    conn = None
    try:
        conn = pymysql.connect(user=user,
                               passwd=password,
                               host=host,
                               port=port,
                               local_infile=1,
                               db=database
                               )
        print(f"Connection established!")
    except Exception as e:
        print(f"Error connecting to the MariaDB Server: {e}")
    return conn


def housing_upload_join_data(conn, year):
  start_date = str(year) + "-01-01"
  end_date = str(year) + "-12-31"

  cur = conn.cursor()
  print('Selecting data for year: ' + str(year))
  cur.execute(f'SELECT pp.price, pp.date_of_transfer, po.postcode, pp.property_type, pp.new_build_flag, pp.tenure_type, pp.locality, pp.town_city, pp.district, pp.county, po.country, po.latitude, po.longitude FROM (SELECT price, date_of_transfer, postcode, property_type, new_build_flag, tenure_type, locality, town_city, district, county FROM pp_data WHERE date_of_transfer BETWEEN "' + start_date + '" AND "' + end_date + '") AS pp INNER JOIN postcode_data AS po ON pp.postcode = po.postcode')
  rows = cur.fetchall()

  csv_file_path = 'output_file.csv'

  # Write the rows to the CSV file
  with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write the data rows
    csv_writer.writerows(rows)
  print('Storing data for year: ' + str(year))
  cur.execute(f"LOAD DATA LOCAL INFILE '" + csv_file_path + "' INTO TABLE `prices_coordinates_data` FIELDS TERMINATED BY ',' OPTIONALLY ENCLOSED by '\"' LINES STARTING BY '' TERMINATED BY '\n';")
  conn.commit()
  
  print('Data stored for year: ' + str(year))




def get_address_area(latitude,longitude):
  """
  this functiontakes the longitude and latitude and returns the building in that 1km by 1km square that have ful addresses.
  if plot is set to true then it plots the data
  """
  #when we say 'around', we mean the square box enclosing the area. not a circular surrounding area
  north = latitude + 0.004
  south = latitude - 0.004
  west = longitude - 0.004
  east = longitude + 0.004
  buildings = ox.geometries_from_bbox(north, south, east, west, {'building': True})

  buildings_df = pd.DataFrame(buildings)
  buildings_df = buildings_df[["building",'addr:city',"addr:housenumber", "addr:street", "addr:postcode",'addr:housename','geometry']]

  #a full address has a street, postcode and a housenumber or name
  addressed = buildings_df[buildings_df['addr:street'].notna() & buildings_df['addr:postcode'].notna() & (buildings_df['addr:housenumber'].notna()|buildings_df['addr:housename'].notna())]


  # 1 degree equates to about 111,111 metres. the area method returns an area in degrees squared
  #to convert to metres squared, multiply it by 111,111 squared. this is not fully accurate away from the equator
  addressed['area'] = addressed['geometry'].apply(lambda x: x.area * 111111 * 111111)

  return(addressed)
