from .config import *
import requests
import pymysql
import osmnx as ox
import matplotlib.pyplot as plt
import warnings
import pandas as pd
import geopandas as gpd


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


def hello_world():
  print("Hello from the data science library!")

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
    pois = ox.geometries_from_bbox(north, south, east, west, tags)


    pois_df = pd.DataFrame(pois)


    #count the occurrences of every tag
    poi_counts = {}
    for tag, tag_values in tags.items():
        if tag in pois_df.columns:
            if isinstance(tag_values, list):
                for value in tag_values:
                    poi_counts[f"{tag}:{value}"] = (pois_df[tag] == value).sum()
            else:
                poi_counts[tag] = pois_df[tag].notnull().sum()
        else:
            poi_counts[tag] = 0

    return poi_counts




def get_address_area(latitude,longitude, plot = False ):
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


  if plot:
    #Plot a map of the area, using pink to mark the buildings with addresses and the ones without.

    #this is the opposite of selectong addressed
    not_addressed = buildings_df[buildings_df['addr:street'].isna() | buildings_df['addr:postcode'].isna() | (buildings_df['addr:housenumber'].isna() & buildings_df['addr:housename'].isna())]
    fig, ax = plt.subplots()
    graph = ox.graph_from_bbox(north, south, east, west)

    # Plot street edges
    nodes, edges = ox.graph_to_gdfs(graph)
    edges.plot(ax=ax, linewidth=1, edgecolor="dimgray")

    ax.set_xlim([west, east])
    ax.set_ylim([south, north])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")

    # Plot buildings
    gdf = gpd.GeoDataFrame(addressed, geometry='geometry')
    gdf2 = gpd.GeoDataFrame(not_addressed, geometry='geometry')
    gdf2.plot(ax=ax, color='grey')
    gdf.plot(ax=ax, color='pink')
    plt.tight_layout()

  return(addressed)
