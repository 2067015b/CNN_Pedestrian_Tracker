import pickle
# import queue
import sys
# from copy import deepcopy
#
# from keras.models import model_from_json
# from skimage import io
import mask.coco as coco
import os
import cv2
import scipy.misc
# import numpy as np
# import matplotlib.pyplot as plt
from condensation.predictFrame import get_predictions_for_frame
from mask.hda import HDAConfig
from Config import *
from Models import Trajectory, Detection
from statistics import Metric
import re_id.cuhk03.model as re_id_m
#
import mask.model as modellib
import mask.hda as hda

# import mask.visualize

trajectories = {}
unassigned = {}


def main(video_path, reid_weights, mask_config='coco', gt=None, use_metrics=False):
    # global FRAME_HEIGHT
    if use_metrics:
        metric = Metric(gt)

    # Root directory of the project
    ROOT_DIR = os.getcwd()

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "mask", "logs")

    # Load the re-identification model and weights
    re_id_model = re_id_m.generate_model()
    re_id_model = re_id_m.compile_model(re_id_model)
    with open(reid_weights, 'rb') as pk:
        re_id_weights = pickle.load(pk)  # load_model("re_id/cuhk03/weights/weights_on_cuhk03_0_0_full.h5")
    re_id_model.set_weights(re_id_weights)


    # Local path to trained weights file
    if mask_config == 'coco':
        MODEL_PATH = os.path.join(ROOT_DIR, 'mask', 'weights', COCO_WEIGHTS_PATH)

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
        model.load_weights(MODEL_PATH, by_name=True)

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

    elif mask_config == 'hda':

        class InferenceConfig(hda.HDAConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        model.load_weights("mask/weights/hda.h5", by_name=True)

        class_names = ['BG','person']


    # with open(RE_ID_MODEL_PATH, 'r') as f:
    #     model_json = f.read()
    #     # load model from json
    #     re_id_model = model_from_json(model_json)
    # load weights
    # re_id_model.load_weights(RE_ID_WEIGHTS_PATH)


    sequence = cv2.VideoCapture(video_path)
    if (sequence.isOpened() == False):
        print('Input video not found: ' + video_path, flush=True)
        sys.exit(1)

    frame = 1
    ret, image = sequence.read()

    rescale_factor = float(FRAME_WIDTH) / image.shape[1]
    FRAME_HEIGHT = int(image.shape[0] * rescale_factor)

    # Run the detection on initial frame
    results_0 = model.detect([image], verbose=1)

    # print("The results format: {}".format(results_0))
    trajectories[str(frame)] = []
    for detection, cls in zip(results_0[0]['rois'], results_0[0]['class_ids']):
        if class_names[cls] == 'person':
            new_traj = Trajectory(frame, detection)
            print(detection, flush=True)
            trajectories[str(frame)].append(new_traj)

    while (frame <= sequence.get(cv2.CAP_PROP_FRAME_COUNT) - FRAME_STEP):

        # Read the next frame
        frame += FRAME_STEP
        sequence.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, next_image = sequence.read()

        trajectories[str(frame)] = []

        # Run the detection on the next frame
        results_1 = model.detect([next_image], verbose=1)
        predictions = get_predictions_for_frame(image, next_image, trajectories[str(frame - FRAME_STEP)])

        for trajectory in trajectories[str(frame - FRAME_STEP)]:
            coords = trajectory.get_detections()
            print("GENERATING RECTANGLES: {}".format(coords), flush=True)
            cv2.rectangle(image, (coords[1], coords[0]), (coords[3], coords[2]), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(image, str(trajectory.id), (coords[1], coords[0]), font, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        for _, det in predictions:
            for prediction in det:
                cv2.circle(image, (int(prediction[1]), int(prediction[0])), 4, (255, 0, 0), 1)
        frame_name = "E:\\Uni\\Level_5_Project\\Tracker\\data\\green\\validation_1_{}.jpg".format(frame)
        cv2.imwrite(frame_name, image)

        candidate_detections = []
        for detection, cls in zip(results_1[0]['rois'], results_1[0]['class_ids']):
            if class_names[cls] == 'person':
                candidate_detections.append(Detection(detection))

        # Match detections in frame 0 with detections in frame 1
        i = 0
        for det0_coords, preds in predictions:
            # print("preds: {}".format(preds), flush=True)
            # Compute the area for each preds
            area_coords = get_area(preds)
            # Array to hold the indices of detections_1 that fall within this area
            # Loop through detections in frame 1
            for detection_1 in candidate_detections[:]:
                det1_coords = detection_1.get_coordinates()
                centroid = get_centroid(det1_coords)

                if centroid[0] >= area_coords[0] and centroid[0] <= area_coords[2] and centroid[1] >= area_coords[1] and \
                                centroid[1] <= area_coords[3]:
                    # print("Detection_1: {}".format(det1_coords), flush=True)

                    # plt.imshow(image[det1_coords[0]:det1_coords[2], det1_coords[1]:det1_coords[3], :])
                    # plt.show()
                    similarity = re_id_m.get_prediction(re_id_model, image[det0_coords[0]:det0_coords[2],
                                                                     det0_coords[1]:det0_coords[3], :],
                                                        next_image[det1_coords[0]:det1_coords[2],
                                                        det1_coords[1]:det1_coords[3], :])
                    scipy.misc.imsave(str(det0_coords) + "_1.jpg", image[det0_coords[0]:det0_coords[2],
                                                                     det0_coords[1]:det0_coords[3], :])
                    scipy.misc.imsave(str(det1_coords) + "_2.jpg", next_image[det1_coords[0]:det1_coords[2],
                                                                     det1_coords[1]:det1_coords[3], :])
                    print(similarity)
                    if similarity >= SIMILARITY_THRESHOLD:
                        print("LENGTH OF CANDIDATE_DETECTIONS: {}".format(len(candidate_detections)), flush=True)
                        candidate_detections.remove(detection_1)
                        print("LENGTH OF CANDIDATE_DETECTIONS: {}".format(len(candidate_detections)), flush=True)

                        updated_trajectory = trajectories[str(frame - FRAME_STEP)].pop(i)
                        i -= 1
                        updated_trajectory.add_detection(frame, det1_coords)
                        trajectories[str(frame)].append(updated_trajectory)

                        break
            i += 1
        print("Finished initial matching.", flush=True)
        # Try to match future detections with ones from previous frames
        for detection_1 in candidate_detections[:]:
            det1_coords = detection_1.get_coordinates()
            current_frame = frame - 2 * FRAME_STEP
            matched = False
            while current_frame >= max(frame - 5 * FRAME_STEP, 1) and not matched:

                i = 0
                for trajectory in trajectories[str(current_frame)]:

                    sequence.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, previous_detection = sequence.read()

                    det0_coords = trajectory.get_detections(frame=current_frame)

                    # TODO: Check for the return value - throw error

                    similarity = re_id_m.get_prediction(re_id_model, previous_detection[det0_coords[0]:det0_coords[2],
                                                                     det0_coords[1]:det0_coords[3], :], next_image[det1_coords[0]:det1_coords[2],
                                                                     det1_coords[1]:det1_coords[3], :])
                    print(similarity)
                    if similarity >= SIMILARITY_THRESHOLD:
                        candidate_detections.remove(detection_1)

                        updated_trajectory = trajectories[str(current_frame)].pop(i)
                        i -= 1
                        updated_trajectory.add_detection(frame, det1_coords)
                        trajectories[str(frame)].append(updated_trajectory)

                        matched = True
                        break

                    i += 1
                current_frame -= FRAME_STEP

        # Treat the rest of the detections as new and generate a trajectory object
        for detection_1 in candidate_detections:
            det1_coords = detection_1.get_coordinates()
            print("Format of det1_coords: {}".format(det1_coords), flush=True)

            # Check if the detection is at the edge of the screen only then create the new trajectory
            centroid = get_centroid(det1_coords)
            if (centroid[0] <= FRAME_WIDTH * FRAME_BOUNDARY or centroid[
                0] >= FRAME_WIDTH - FRAME_WIDTH * FRAME_BOUNDARY) and \
                    (centroid[1] <= FRAME_HEIGHT * FRAME_BOUNDARY or centroid[
                        1] >= FRAME_HEIGHT - FRAME_HEIGHT * FRAME_BOUNDARY):
                trajectories[str(frame)].append(Trajectory(frame, det1_coords))

        # cv2.imshow('output', image)
        # cv2.waitKey(delay=1)

        image = next_image
        # assert len(candidate_detections) == 0
        dets = []
        for trajectory in trajectories[str(frame)]:
            dets.append(trajectory.get_detections(frame))
        metric.log(frame, dets)

        print(trajectories, flush=True)
        # cv2.destroyAllWindows()

    with open("E:\\Uni\\Level_5_Project\\Tracker\\data\\green\\validation_1.pickle", 'wb') as output_file:
        pickle.dump(trajectories, output_file)
    metric.print_stats()


def get_centroid(detection):
    return (detection[0] + ((detection[2] - detection[0]) / 2), detection[1] + ((
                                                                                    detection[3] - detection[1]) / 2))


def get_area(predictions):
    min_y = FRAME_WIDTH
    max_y = 0

    min_x = FRAME_HEIGHT
    max_x = 0

    # Detect the boundary
    for prediction in predictions:
        if prediction[0] > max_x:
            max_x = prediction[0]
        if prediction[0] < min_x:
            min_x = prediction[0]
        if prediction[1] > max_y:
            max_y = prediction[1]
        if prediction[1] < min_y:
            min_y = prediction[1]

        # Pad area by 10%
        x_padding = (max_x - min_x) * 0.5
        y_padding = (max_y - min_y) * 0.5

        min_x, max_x = min_x - x_padding, max_x + x_padding
        min_y, max_y = min_y - y_padding, max_y + y_padding

    return (min_x, min_y, max_x, max_y)


if __name__ == "__main__":

    # main(*sys.argv[1:], reid_weights="E:\\Uni\\Level_5_Project\\Tracker\\re_id\\cuhk03\\weights\\weights_on_cuhk03_0_0.pickle",)
    main("E:\\Uni\\Level_5_Project\\Tracker\\data\\green\\validation_1.avi",
         reid_weights="E:\\Uni\\Level_5_Project\\Tracker\\re_id\\cuhk03\\weights\\weights_on_cuhk03_0_0.pickle",
         gt="E:\\Uni\\Level_5_Project\\Tracker\\data\\green\\validation_1\\allD.txt",
         use_metrics=True)
