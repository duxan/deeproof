"""
Mask R-CNN
Train on the toy Buildings dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 buildings.py train --dataset=/path/to/buildings/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 buildings.py train --dataset=/path/to/buildings/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 buildings.py train --dataset=/path/to/buildings/dataset --weights=imagenet

    # Apply color splash to an image
    python3 buildings.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 buildings.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime

import imageio
import imgaug
import numpy as np
import skimage.draw
from imgaug import augmenters as iaa

# Root directory of the project
ROOT_DIR = os.getcwd()
if ROOT_DIR.endswith("samples/buildings"):
    # Go up two levels to the repo root
    ROOT_DIR = os.path.dirname(os.path.dirname(ROOT_DIR))

# Import Mask RCNN
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class BuildingConfig(Config):
    """Configuration for training on the building dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "building"

    # backbone, and we start from imagenet resnet101 weights
    BACKBONE = "resnet101"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 8 + 2 + 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.75

    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    TRAIN_ROIS_PER_IMAGE = 120
    IMAGE_PADDING = True

    LEARNING_RATE = 0.005


############################################################
#  Dataset
############################################################


class BuildingDataset(utils.Dataset):

    def load_building(self, dataset_dir, subset):
        """Load a subset of the Building dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        self.add_class("building", 1, "flat")
        self.add_class("building", 2, "dome")
        self.add_class("building", 3, "tree")
        for i, cl in enumerate(directions):
            self.add_class("building", i + 4, cl)

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        images = [name for name in os.listdir(os.path.join(dataset_dir, "images")) if name.endswith(".jpg")]
        for img in images:
            self.add_image(
                "building",
                image_id=img,  # use file name as a unique image id
                path=os.path.join(dataset_dir, "images", img),
                width=512, height=512)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]

        mask_path = info['path'].replace("images", "masks")
        mask_files = [name for name in os.listdir(mask_path)
                      if os.path.isfile(os.path.join(mask_path, name)) and name.endswith(".jpg")]
        mask_count = len(mask_files)

        mask = np.zeros([info["height"], info["width"], mask_count], dtype=np.uint8)
        class_ids = np.zeros(mask_count, np.int32)

        for i, p in enumerate(mask_files):
            mask_img = skimage.io.imread(os.path.join(mask_path, p), as_gray=True)
            rr, cc = np.nonzero(mask_img)
            class_name = p.split(".")[0].split("-")[-1]
            class_ids[i] = self.class_name_map[class_name]
            mask[rr, cc, i] = 1
            print_mask = mask[:, :, i]
            #imgaug.imshow(print_mask)

        # Return mask, and array of class IDs of each instance
        return mask, class_ids

    def load_building_json_masks(self, dataset_dir, subset):
        """Load a subset of the Building dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes
        directions = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        self.add_class("building", 1, "flat")
        self.add_class("building", 2, "dome")
        self.add_class("building", 3, "tree")
        for i, cl in enumerate(directions):
            self.add_class("building", i + 4, cl)

        # Train or validation dataset?
        assert subset in ["train", "val", "test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # We mostly care about the x and y coordinates of each region
        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]
            region_attributes = [r['region_attributes'] for r in a['regions'].values()]
            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "building",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons, region_attributes=region_attributes)

    def load_mask_from_polygons(self, image_id, with_class=True):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        mask_count = len(info["polygons"])
        class_ids = np.zeros(mask_count, np.int32)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to the attribute
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            class_name = info['region_attributes'][i]["building"].strip()
            class_ids[i] = self.class_name_map[class_name]
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance
        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "building":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = BuildingDataset()
    dataset_train.load_building(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = BuildingDataset()
    dataset_val.load_building(args.dataset, "val")
    dataset_val.prepare()

    # Training - Stage 1
    # Heads only
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                layers='heads')

    # Training - Stage 2
    # Finetune layers 4 and up
    print("Fine tune layers 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=6,
                layers='4+')

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 50,
                epochs=10,
                layers='all')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # We're treating all instances as one, so collapse the mask into one layer
    mask = (np.sum(mask, -1, keepdims=True) >= 1)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[0] > 0:
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)
        r = r[0]
        if len(r["masks"]) == 0:
            print("No segments detected, exiting without image output")
            return False
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect buildings.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/building/dataset/",
                        help='Directory of the building dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BuildingConfig()
    else:
        class InferenceConfig(BuildingConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()[1]
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "snapshots/Use 'train' or 'splash'".format(args.command))
