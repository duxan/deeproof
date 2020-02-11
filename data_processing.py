import json
import os
import random

import imageio
import numpy as np
import imutils
import copy
import colorsys
import cv2
import math

import imgaug
from imgaug import augmenters as iaa, SegmentationMapsOnImage

from mrcnn import visualize
from mrcnn.buildings import BuildingDataset
from mrcnn.utils import extract_bboxes


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
            x, y = rotate((256, 256), (x, y), angle)
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

    M = cv2.getRotationMatrix2D((256, 256), angle, 1)
    points = np.stack((p['all_points_x'], p['all_points_y'], np.ones(len(p['all_points_x']))))
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


def between(a, x, b):
    return a < x <= b


def get_direction_label_16(degree):
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


def get_direction_label_8(degree):
    if between(337.25, degree, 360) or between(0, degree, 22.5) or degree == 0:
        return "N"
    elif between(22.5, degree, 67.5):
        return "NE"
    elif between(67.5, degree, 112.5):
        return "E"
    elif between(112.5, degree, 157.5):
        return "SE"
    elif between(157.5, degree, 202.5):
        return "S"
    elif between(202.5, degree, 247.5):
        return "SW"
    elif between(247.5, degree, 292.5):
        return "W"
    elif between(292.5, degree, 337.5):
        return "NW"
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

            image = cv2.imread(os.path.join(infolder, filename))

            for angle in np.arange(0, 360, rotation):

                mod_key = k + "." + str(angle)
                mod_filename = obj["filename"][:-3] + str(angle) + ".jpg"

                out[mod_key] = copy.deepcopy(obj)
                out[mod_key]["filename"] = mod_filename

                out[mod_key]["width"] = 512
                out[mod_key]["height"] = 512
                out[mod_key]["zoom"] = 20

                # rotate image
                rotated = imutils.rotate(image, angle)

                for r in obj["regions"].keys():
                    shape = out[mod_key]["regions"][r]["shape_attributes"]
                    rotate_poly = rotate_polygon(shape, angle)

                    out[mod_key]["regions"][r]["shape_attributes"]["all_points_x"] = list(
                        map(lambda x: int(x), rotate_poly["all_points_x"]))
                    out[mod_key]["regions"][r]["shape_attributes"]["all_points_y"] = list(
                        map(lambda x: int(x), rotate_poly["all_points_y"]))
                    try:
                        label = get_direction_label_8(
                            (int(out[mod_key]["regions"][r]["region_attributes"]["building"]) - angle) % 360)
                    except:

                        label = out[mod_key]["regions"][r]["region_attributes"]["building"]
                        if label not in ["tree", "flat"]:
                            print("ERROR: ", label, filename)
                            exit()

                    out[mod_key]["regions"][r]["region_attributes"]["building"] = label

                    # write image
                cv2.imwrite(outfolder + "/" + mod_filename, rotated)

        files = [f for f in os.listdir(outfolder) if ".jpg" in f]

        print("Number of augmented files: ", len(files))

        with open(outfolder + "/via_region_data.json", "w") as fp:
            json.dump(out, fp)


def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = np.where(mask == 1, image[:, :, c] * (1 - alpha) + alpha * color[c] * 255, image[:, :, c])
    return image


def random_colors(N, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def imgaug_augmentation(infolder, outfolder, imagefolder, aug_name, seq, write_originals=False):
    if not os.path.exists(os.path.join(outfolder, imagefolder)):
        os.makedirs(os.path.join(outfolder, imagefolder, "images"))
        os.makedirs(os.path.join(outfolder, imagefolder, "masks"))

    dataset = BuildingDataset()
    dataset.load_building_json_masks(infolder, imagefolder)
    dataset.prepare()

    print(f"Dataset {imagefolder.upper()} contains {dataset.num_classes} classes in {dataset.num_images} images")
    images = [dataset.load_image(img_id) for img_id in dataset.image_ids]
    image_names = [dataset.source_image_link(img_id).split("/")[-1] for img_id in dataset.image_ids]
    mask_class_ids = [dataset.load_mask_from_polygons(img_id) for img_id in dataset.image_ids]
    class_ids = [mask_id[1] for mask_id in mask_class_ids]  # store for saving masks later
    class_names_map = dataset.class_name_map
    segmaps = [SegmentationMapsOnImage(mask_id[0], shape=(512, 512)) for mask_id in mask_class_ids]

    if write_originals:
        print("Writing original images and masks [can take awhile] ...")
        for n in image_names:
            os.makedirs(os.path.join(outfolder, imagefolder, "masks", n), exist_ok=True)

        for img_name, img, classes, segs in zip(image_names, images, class_ids, segmaps):
            imageio.imwrite(os.path.join(outfolder, imagefolder, "images", img_name), img)
            for i, cls in enumerate(classes):
                cls = {v: k for k, v in class_names_map.items()}[cls]
                imageio.imwrite(os.path.join(outfolder, imagefolder, "masks",
                                             img_name, f"segment-{i}_class-{cls}.jpg"), segs.draw()[i])

    print(f"Augmenting images: {aug_name.upper()} [can take awhile] ...")
    for n in image_names:
        n = n + "_" + aug_name + ".jpg"
        os.makedirs(os.path.join(outfolder, imagefolder, "masks", n), exist_ok=True)

    imgs_aug, segmaps_aug = seq(images=images, segmentation_maps=segmaps)
    for img_name, img, classes, segs in zip(image_names, imgs_aug, class_ids, segmaps_aug):
        img_name = img_name + "_" + aug_name + ".jpg"
        imageio.imwrite(os.path.join(outfolder, imagefolder, "images", img_name), img)
        for i, cls in enumerate(classes):
            cls = {v: k for k, v in class_names_map.items()}[cls]
            imageio.imwrite(os.path.join(outfolder, imagefolder, "masks", img_name,
                                         f"segment-{i}_class-{cls}.jpg"), segs.draw()[i])

    # visualize_imgaug(images[0], segmaps[0], imgs_aug[0], segmaps_aug[0])


def visualize_imgaug(img, segmap, img_aug, segmap_aug, segment_idx=None):
    img_with_seg = segmap.draw_on_image(img)
    if not segment_idx:
        segment_idx = random.randint(0, len(img_with_seg))

    cells = [img, img_with_seg[segment_idx], img_aug, segmap_aug.draw_on_image(img_aug)[segment_idx]]
    imgaug.show_grid(cells, cols=2)


if __name__ == "__main__":
    # inputfolder = "./data/deeproof/"
    # outfolder = "./images/deeproof/"

    # augment the data
    # OLD code - from angles to classes, saves Via annotaion file per folder + does only rotation
    # rotates the image and replaces azimuth values with labels 
    # imagefolder = "train"
    # data_augmentation(inputfolder + "/train", outfolder + "/train", rotation=360)

    # imagefolder = "val"
    # data_augmentation(inputfolder + "/val", outfolder + "/val", rotation=360)

    # imagefolder = "test"
    # data_augmentation(inputfolder + "/test", outfolder + "/test", rotation=360)

    # example complex augmentation
    seq = iaa.Sequential([
        iaa.Dropout([0.0, 0.02]),  # drop 5% or 20% of all pixels
        iaa.Sharpen((0.0, 0.1)),  # sharpen the image
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        iaa.CropAndPad(percent=(-0.25, 0.25)),
        iaa.GaussianBlur(sigma=(0.0, 1.5)),
        iaa.ElasticTransformation(alpha=991, sigma=8)  # apply water effect (affects segmaps)
    ], random_order=True)

    inputfolder = "./images/deeproof/"  # via annotations, class names not angles, 8 directions
    outfolder = "./images/deeproof-aug/"  # masks as files, aug with imgaug

    # TRAIN
    imagefolder = "train"
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Affine(rotate=(-45, 45)),
                        aug_name="rotate", write_originals=True)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.CropAndPad(percent=(-0.25, 0.25)),
                        aug_name="crop", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Sharpen((0.1, 0.5)),
                        aug_name="sharpen", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.GaussianBlur(sigma=(0.1, 2.0)),
                        aug_name="blur", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.ElasticTransformation(alpha=17, sigma=8),
                        aug_name="water", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Dropout([0.05, 0.2]),
                        aug_name="dropout", write_originals=False)
    num_images = len([name for name in os.listdir(os.path.join(outfolder, imagefolder, "images"))])
    num_masks = len([name for name in os.listdir(os.path.join(outfolder, imagefolder, "masks"))
                     if os.path.isfile(os.path.join(os.path.join(outfolder, imagefolder, "masks"), name))])

    print(f"Total number of images in {imagefolder} is {num_images}")
    print(f"Total number of masks in {imagefolder} is {num_images}")

    # VAL
    imagefolder = "val"
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Affine(rotate=(-45, 45)),
                        aug_name="rotate", write_originals=True)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.CropAndPad(percent=(-0.25, 0.25)),
                        aug_name="crop", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Sharpen((0.1, 0.5)),
                        aug_name="sharpen", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.GaussianBlur(sigma=(0.1, 2.0)),
                        aug_name="blur", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.ElasticTransformation(alpha=17, sigma=8),
                        aug_name="water", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Dropout([0.05, 0.2]),
                        aug_name="dropout", write_originals=False)
    num_images = len([name for name in os.listdir(os.path.join(outfolder, imagefolder, "images"))])
    num_masks = len([name for name in os.listdir(os.path.join(outfolder, imagefolder, "masks"))
                     if os.path.isfile(os.path.join(os.path.join(outfolder, imagefolder, "masks"), name))])

    print(f"Total number of images in {imagefolder} is {num_images}")
    print(f"Total number of masks in {imagefolder} is {num_images}")

    # TEST
    imagefolder = "test"
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Affine(rotate=(-45, 45)),
                        aug_name="rotate", write_originals=True)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.CropAndPad(percent=(-0.25, 0.25)),
                        aug_name="crop", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Sharpen((0.1, 0.5)),
                        aug_name="sharpen", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.GaussianBlur(sigma=(0.1, 2.0)),
                        aug_name="blur", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.ElasticTransformation(alpha=17, sigma=8),
                        aug_name="water", write_originals=False)
    imgaug_augmentation(inputfolder, outfolder, imagefolder, seq=iaa.Dropout([0.05, 0.2]),
                        aug_name="dropout", write_originals=False)
    num_images = len([name for name in os.listdir(os.path.join(outfolder, imagefolder, "images"))])
    num_masks = len([name for name in os.listdir(os.path.join(outfolder, imagefolder, "masks"))
                     if os.path.isfile(os.path.join(os.path.join(outfolder, imagefolder, "masks"), name))])

    print(f"Total number of images in {imagefolder} is {num_images}")
    print(f"Total number of masks in {imagefolder} is {num_images}")
