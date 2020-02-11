import matplotlib
from matplotlib import patches
from matplotlib.patches import Polygon
matplotlib.use('Agg')

import inspect
import os
import random
import sys

import requests
from flask import Flask, render_template, request, url_for
from flask_googlemaps import GoogleMaps, Map
from skimage.measure import find_contours

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import numpy as np
import mrcnn.model as modellib
from mrcnn.buildings import BuildingConfig
from mrcnn import utils
from mrcnn.visualize import random_colors, apply_mask
import matplotlib.pyplot as plt
import skimage.draw

API_ENDPOINT = "https://maps.googleapis.com/maps/api/staticmap"
STATIC_PARAMS = "maptype=satellite&format=jpg&zoom=20&size=512x512"
API_KEY = os.getenv('GOOGLE_API_KEY')
MODEL = "../logs/5/mask_rcnn_building_0010.h5"


class InferenceConfig(BuildingConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.65


config = InferenceConfig()
model = modellib.MaskRCNN(mode="inference", config=config, model_dir="../logs/")
model.load_weights(MODEL, by_name=True)
model.keras_model._make_predict_function()

app = Flask(__name__)
GoogleMaps(app, key=API_KEY)


@app.route('/', methods=['GET', 'POST'])
def index():
    json_data = {
        'user': {
            'x': 44.7818731,
            'y': 20.477353
        }
    }

    user_location = (json_data['user']['x'], json_data['user']['y'])

    map = Map(
        identifier="map", varname="map",
        lat=user_location[0], lng=user_location[1],
        zoom=17,
        setTilt=0,
        maptype="SATELLITE",
        zoom_control=True,
        maptype_control=False,
        scale_control=False,
        streetview_control=False,
        rotate_control=False,
        fullscreen_control=False,
        scroll_wheel=False,
        collapsible=False,
        center_on_user_location=True
    )

    return render_template('map.html', map=map, app_title="Rooftop instance segmentation")


@app.route('/get_map_coords', methods=['POST'])
def get_map_coords():
    payload = request.form
    point = ",".join([payload['lat'], payload['lng']])
    url = f"{API_ENDPOINT}?{STATIC_PARAMS}&center={point}&key={API_KEY}"
    img = requests.get(url)
    with open(f'static/downloaded.jpg', 'wb') as f:
        f.write(img.content)
    f.close()

    return url_for("static", filename='downloaded.jpg')


@app.route('/score', methods=['GET'])
def score():
    image_path = "static/downloaded.jpg"
    image = skimage.io.imread(image_path)

    r = model.detect([image], verbose=1)
    r = r[0]

    mask, class_ids, scores = r['masks'], r['class_ids'], r['scores']
    bbox = utils.extract_bboxes(mask)
    classes = ["BG", "flat", "dome", "tree", "N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    display_instances(image, bbox, mask, class_ids, classes, scores)

    return url_for("static", filename='scored.jpg')


def display_instances(image, boxes, masks, class_ids, class_names, scores=None):
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    _, ax = plt.subplots(1, figsize=(6.4, 6.4))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              alpha=0.7, linestyle="dashed",
                              edgecolor=color, facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        score = scores[i] if scores is not None else None
        label = class_names[class_id]
        x = random.randint(x1, (x1 + x2) // 2)
        caption = "{} {:.3f}".format(label, score) if score else label
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        mask = masks[:, :, i]
        masked_image = apply_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))

    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0, 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig("static/scored.jpg", dpi=80, facecolor='w', edgecolor='k', bbox_inches='tight', pad_inches = 0)


if __name__ == '__main__':
    app.run(debug=True)
