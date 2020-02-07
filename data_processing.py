import json
import os
import random
import numpy as np
import imutils
import copy
import colorsys
from OSMPythonTools.api import Api

import cv2
import math


api = Api()


def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    
    return round(qx), round(qy)


def rotate_poly(polygons, angle):
    poly = []
    for p in polygons:
        obj = {}
        all_points_y = []
        all_points_x = []
        for y, x in zip(p['all_points_y'], p['all_points_x']):
            x, y = rotate((256, 256), (x,y), angle)
            x = max(0, x)
            x = min(x, 512)
            y = max(0, y)
            y = min(x, 512)
            all_points_y.append(y)
            all_points_x.append(x)
        
        obj["all_points_y"] = all_points_y
        obj["all_points_x"] = all_points_x
        poly.append(obj)
    return poly


def rotate_polygon(p, angle):
    points = []
    for x, y in zip(p['all_points_x'], p['all_points_y']):
        points.append([x, y])
    points = np.array(points)
    
    M = cv2.getRotationMatrix2D((256,256),angle,1)
    points =  np.stack((p['all_points_x'], p['all_points_y'], np.ones(len(p['all_points_x']))))
    src = np.round(M.dot(points)).astype(int)
        
    src[src > 512] = 512
    src[src < 0] = 0
    obj = {"all_points_x": src[0], "all_points_y": src[1]}
    return obj


def rotate_polygons(polygons, angle):
    poly = []
    for p in polygons:
        p = rotate_polygon(p, angle)
        poly.append(p)
    return poly


def between(a, x,  b):
    return a < x <= b


def get_direction_label(degree):
    
    if between(348.75, degree, 360) or between(0, degree, 11.25) or degree == 0:
        return "N"
    elif between(11.25, degree, 33.75):
        return "NNE"
    elif between(33.75, degree, 56.25):
        return "NE"
    elif between(56.25, degree, 78.75):
        return "ENE"
    elif between(78.75, degree, 101.25):
        return "E"
    elif between(101.25, degree, 123.75):
        return "ESE"
    elif between(123.75, degree, 146.25):
        return "SE"
    elif between(146.25, degree, 168.75):
        return "SSE"
    elif between(168.75, degree, 191.25):
        return "S"
    elif between(191.25, degree, 213.75):
        return "SSW"
    elif between(213.75, degree, 236.25):
        return "SW"
    elif between(236.25, degree, 258.75):
        return "WSW"
    elif between(258.75, degree, 281.25):
        return "W"
    elif between(281.25, degree, 303.75):
        return "WNW"
    elif between(303.75, degree, 326.25):
        return "NW"
    elif between(326.25, degree, 348.75):
        return "NNW"
    else:
        print("error.", degree)
        exit()


def data_augmentation(infolder, outfolder, rotation=30):
    
    # augment training data
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    out = {}
    with open(infolder + "/via_region_data.json", 'r') as fp:
        data = json.load(fp)
        for k in data.keys():
            obj = data[k].copy()
            filename = obj["filename"]
            # get existing polygon
            polygons = [r['shape_attributes'] for r in obj['regions'].values()]

            image = cv2.imread( os.path.join(infolder, filename) )

            for angle in np.arange(0, 360, rotation):
                
                mod_key = k + "." + str(angle)
                mod_filename = obj["filename"][:-3] + str(angle) + ".jpg"

                out[mod_key] = copy.deepcopy(obj)
                out[mod_key]["filename"]=mod_filename                
                
                out[mod_key]["width"] = 512
                out[mod_key]["height"] = 512
                out[mod_key]["zoom"] = 20
                
                # rotate image
                rotated = imutils.rotate(image, angle)
                
                for r in obj["regions"].keys():
                    shape =  out[mod_key]["regions"][r]["shape_attributes"]
                    rotate_poly = rotate_polygon(shape, angle)
                    
                    out[mod_key]["regions"][r]["shape_attributes"]["all_points_x"] = list(map(lambda x: int(x), rotate_poly["all_points_x"]))
                    out[mod_key]["regions"][r]["shape_attributes"]["all_points_y"] = list(map(lambda x: int(x), rotate_poly["all_points_y"]))
                    try:
                        label = get_direction_label((int(out[mod_key]["regions"][r]["region_attributes"]["building"])-angle)  % 360  )
                    except:
                        
                        label = out[mod_key]["regions"][r]["region_attributes"]["building"]
                        if label not in ["tree", "flat"]:
                            print("ERROR: ", label, filename)
                            exit()

                    out[mod_key]["regions"][r]["region_attributes"]["building"] = label 
                
                # write image
                cv2.imwrite(outfolder+"/"+mod_filename, rotated)

        files = [ f for f in os.listdir(outfolder) if ".jpg" in f]
        flat_files = [f for f in files if "FLAT" in f]
        
        print("Number of augmented files: ", len(files))

        with open(outfolder + "/via_region_data.json", "w") as fp:
            json.dump(out, fp)
    

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,image[:, :, c] *(1 - alpha) + alpha * color[c] * 255,image[:, :, c])
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


if __name__ == "__main__":
    inputfolder = "./data/deeproof/"
    outfolder = "./data/deeproof-aug/"
    
    # augment the data
    # rotates the image and replaces azimuth values with labels 
    imagefolder = "train"
    data_augmentation(inputfolder + "/train", outfolder + "/train", rotation=360)
    
    imagefolder = "val"
    data_augmentation(inputfolder + "/val", outfolder + "/val", rotation=360)
    
    imagefolder = "test"
    data_augmentation(inputfolder + "/test", outfolder + "/test", rotation=360)
