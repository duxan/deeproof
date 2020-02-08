for f in data/deeproof/train/*.jpg
do
 echo "Processing $f"
 python mrcnn/buildings.py splash --weights=logs/mask_rcnn_building_0052.h5 --image=$f
done
