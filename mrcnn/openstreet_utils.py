import math 
import os 
import pandas as pd 
from OSMPythonTools.api import Api
from shapely.geometry import Polygon
from PIL import Image, ImageDraw 
api = Api()
# input_file = "/home/stephenlee/Documents/workspace/solaropia/solaropia-fetch/patriotproperty/patriotPropertyRooftop/output/framingham/framingham_sunroof_old.csv"
# df = pd.read_csv(input_file)

# input_file = "/home/stephenlee/Documents/workspace/solaropia/solaropia-fetch/code/cities/St.Petersburgh_buildings_sunroof.csv"
# df_flo= pd.read_csv(input_file)

input_file = "/home/stephenlee/Documents/workspace/solaropia/solaropia-fetch/code/cities/all-citiesv1.csv"
input_file = "/home/stephenlee/Documents/workspace/solaropia/solaropia-fetch/code/cities/sunroof_cities.csv"
df= pd.read_csv(input_file)


def get_latlon(wayid, city="framingham"):
    df_sel = df[df["wayid"]==wayid].iloc[0]
    lat = float(df_sel["lat"])
    lon = float(df_sel["lon"])

    return lat, lon 

# def get_latlon(wayid, city="framingham"):
#     print(wayid)
#     if city == "framingham":
#         df_sel = df[df["wayid"]==wayid].iloc[0]
#     else:
#         df_sel = df_flo[df_flo["wayid"]==wayid].iloc[0]

#     lat = float(df_sel["lat"])
#     lon = float(df_sel["lon"])

#     return lat, lon 


def get_lat_lon_point(lat, lon, point_lat, pointlon, width=512, height=512, zoom=20):
    parallel_multiplier = math.cos(lat * math.pi / 180)
    degrees_per_pixel_x = 360 / math.pow(2, zoom + 8)
    degrees_per_pixel_y = 360 / math.pow(2, zoom + 8) * parallel_multiplier
    
    y = (lat - point_lat)/degrees_per_pixel_y + (height/2.)
    x = (pointlon - lon)/degrees_per_pixel_x + (width/2.)

    return (x, y)

def get_polygon_mask(wayid, centerlat, centerlon, width=512, height=512, zoom=20):
    # get lat lat lon`
    way = api.query('way/{}'.format(wayid))
    nodes = way.nodes()
    poly = [ get_lat_lon_point(centerlat, centerlon, float(node.lat()), float(node.lon())) for node in nodes]
    
    return poly

def load_building_mask(wayid):
    # extract the way id
    df_sel = df[df["wayid"]==wayid]
    if len(df_sel) > 0:
        lat = float(df_sel["lat"])
        lon = float(df_sel["lon"])
        poly = get_polygon_mask(wayid, lat, lon)
    else:
        raise Exception("Way ID not found")

    return poly

def apply_mask(image, mask):
    """Apply the given mask to the image.
    """
    image = Image.fromarray(image*mask)
    
    return image

if __name__== "__main__":
    


    print(load_building_mask(213264090))