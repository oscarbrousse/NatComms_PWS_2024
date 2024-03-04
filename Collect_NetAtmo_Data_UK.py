# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 10:56:17 2021

@author: oscar
"""

import patatmo as patatmo
import pandas as pd
import fiona
import geopandas as gpd
from shapely.geometry import Polygon
import numpy as np
from time import sleep
from pathlib import Path
import json


datadir = r'[YOUR DIRECTORY]/data/' ### Put the list of stations in the appropriate directory 
date_range = pd.date_range(start='01-05-2022 00:30:00+00:00', end='01-09-2022 00:30:00+00:00', 
                           freq='1H', inclusive='left')

if Path(datadir + 'cred_netatmo_id.txt').is_file():
    credentials = json.loads(open(datadir + 'cred_netatmo_id.txt').read())
else:
    # set up your patatmo connect developer credentials and store them in a txt file
    credentials = {
        "password":"",
        "username":"",
        "client_id":"",
        "client_secret":"",
        "scope": "read_station",
        "grant_type": "password"
    }
    with open(datadir + 'cred_netatmo_id.txt', 'w') as na_cred_file:
         na_cred_file.write(json.dumps(credentials))
  

# create an api client
client = patatmo.api.client.NetatmoClient()

# tell the client's authentication your credentials
client.authentication.credentials = credentials

#%% lat/lon outline for the UK
## Get boundaries from global shapefile
## Downloaded from https://www.data.gov.uk/dataset/0dfd3dab-30ae-4366-995c-70c0864c30a3/local-administrative-units-level-1-december-2015-super-generalised-clipped-boundaries-in-england-and-wales
enwl_lsoa = gpd.read_file(datadir + "UK_Admin_Units/Local_Administrative_Units_Level_1_(December_2015)_Boundaries.shp")
enwl_lsoa.head()

enwl = enwl_lsoa.dissolve()
enwl = enwl.set_crs('epsg:4326')
## Define boundaries of the box in which data will be gathered
llon, llat, ulon, ulat = enwl.geometry.bounds.values.squeeze()

## The Netatmo API automaticly lowers the amount of stations one can access on large scales
## According to colleagues in Germany, moving windows, or tiles, of 0.2°*0.2° permit the acquisition of all stations
lon_tiles = np.append(np.around(np.arange(llon, ulon, 0.2), decimals = 2), np.array(ulon))
lat_tiles = np.append(np.around(np.arange(llat, ulat, 0.2), decimals = 2), np.array(ulat))

list_all_stations = pd.DataFrame(columns=['Lon', 'Lat', 'ID','moduleID', 'index']) 
for x in range(len(lon_tiles)-1):
    for y in range(len(lat_tiles)-1):
        print(lon_tiles[x], lat_tiles[y])
        ### Create a georeferenced polygon of the tile
        poly = Polygon(zip([lon_tiles[x+1], lon_tiles[x+1], lon_tiles[x], lon_tiles[x]], 
                            [lat_tiles[y+1], lat_tiles[y], lat_tiles[y], lat_tiles[y+1]]))
        p = gpd.GeoDataFrame(data = ['tile'], geometry = [poly], crs = enwl.crs)
        ### Check if the tile overlays the UK borders, if not continue loop
        ### Avoids increasing unnecessarily the number of requests to Netatmo server
        if gpd.overlay(enwl, p, how='intersection').empty:
            print('Tile not over land!')
            continue
        
        region = {
            "lat_ne" : lat_tiles[y+1],
            "lat_sw" : lat_tiles[y],
            "lon_ne" : lon_tiles[x+1],
            "lon_sw" : lon_tiles[x],
        }
        # issue the API request
        output = client.Getpublicdata(region = region, filter=True)
        print(len(output.response["body"]))
        
        ### Acquiring the names and places of all stations available in the domain
        stations = output.response["body"]
        ### If there are no stations in the tile jump to the next loop step
        if len(stations) == 0:
            print('No stations in this tile...')
            sleep(10)
            continue

        df = pd.DataFrame(columns=['Lon', 'Lat', 'ID','moduleID', 'index'])
        
        for i in range(0,len(stations)):
            ### Check if the station records temperature:
            list_var = []
            for key, value in stations[i]["measures"].items():
                if value.get('type') is None:
                    continue
                [list_var.append(v) for v in value.get('type')]
            if 'temperature' in list_var:           
                lon = stations[i]["place"]["location"][0]
                lat = stations[i]["place"]["location"][1]
                device = stations[i]["_id"]
                module_id = tuple(stations[i]["measures"].keys())[0]
                new = pd.DataFrame(np.array([[lon, lat, device, module_id]]), 
                                    columns=['Lon', 'Lat', 'ID','moduleID'])
          
                df.loc[i,:] = np.array([[lon, lat, device, module_id,i]])
                ### For an obscure reason the gathering of data includes stations outside of the box
                df_filt = df[(df['Lon'].astype(float) > lon_tiles[x]) & 
                             (df['Lon'].astype(float) < lon_tiles[x+1]) &
                             (df['Lat'].astype(float) > lat_tiles[y]) & 
                             (df['Lat'].astype(float) < lat_tiles[y+1])].copy()
            else:
                print('There were stations, but not with temperature :3')
                
        print(df_filt)
        list_all_stations = pd.concat([list_all_stations, df_filt], ignore_index=True)
        print(list_all_stations.head(), list_all_stations.tail())   
        del df_filt, df, p, poly
        sleep(10)

list_all_stations.to_csv(datadir + 'List_Netatmo_stations_London_ENWL.csv')
