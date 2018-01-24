

import sys
from skimage import io
import mask.coco as coco
import os
import cv2
from condensation.predictFrame import get_predictions_for_frame

import mask.model as modellib
import mask.visualize


FRAME_RATE = 12

def main(argv):
    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "mask", "logs")

    # Local path to trained weights file
    COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'mask', "mask_rcnn_coco.h5")

    videoPath = str(argv[0])

    class InferenceConfig(coco.CocoConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # Load weights trained on MS-COCO
    model.load_weights(COCO_MODEL_PATH, by_name=True)

    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    sequence = cv2.VideoCapture(videoPath)
    if (sequence.isOpened() == False):
        print('Input video not found: ' + videoPath)
        sys.exit(1)

    frame=1
    ret, image = sequence.read()

    while( frame <= sequence.get(cv2.CAP_PROP_FRAME_COUNT)-FRAME_RATE):



        # COCO Class names
        # Run detection
        results = model.detect([image], verbose=1)

        # Visualize results
        r = results[0]

        frame += FRAME_RATE
        sequence.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, next_image = sequence.read()

        predictions = get_predictions_for_frame(image,next_image, r['rois'])

        image = next_image

        print(predictions)





if __name__ == "__main__":
    main(sys.argv[1:])