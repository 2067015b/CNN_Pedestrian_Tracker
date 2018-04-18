import pickle
# import queue
import sys
# from copy import deepcopy
#
# from keras.models import model_from_json
# from skimage import io
import mask.coco as coco
import os
from scipy.misc import imresize
import cv2
import scipy.misc
import numpy as np
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
import mask.uog as uog

# import mask.visualize
START_FRAME = 0
FRAME_HEIGHT = 0
trajectories = {}
unassigned = {}


def main(video_path, reid_weights, mask_config='uog', gt=None, use_metrics=True):
    global FRAME_HEIGHT
    global FRAME_WIDTH

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

    elif mask_config == 'uog':

        class InferenceConfig(uog.UOGConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1

        config = InferenceConfig()
        config.display()

        # Create model object in inference mode.
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
        model.load_weights("mask/weights/uog.h5", by_name=True)

        class_names = ['BG','person']


    sequence = cv2.VideoCapture(video_path)
    if (sequence.isOpened() == False):
        print('Input video not found: ' + video_path, flush=True)
        sys.exit(1)

    frame = START_FRAME
    sequence.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ret, image = sequence.read()

    # rescale_factor = float(FRAME_WIDTH) / image.shape[1]
    # FRAME_HEIGHT = int(image.shape[0] * rescale_factor)
    FRAME_WIDTH = image.shape[1]
    FRAME_HEIGHT = image.shape[0]

    # image = imresize(image,(FRAME_HEIGHT,FRAME_WIDTH,3))
    # Run the detection on initial frame
    results_1 = model.detect([image], verbose=1)

    # print("The results format: {}".format(results_0))
    trajectories[str(frame)] = []
    for detection, cls in zip(results_1[0]['rois'], results_1[0]['class_ids']):

        # print(class_names[cls])
        if class_names[cls] == 'person':
            new_traj = Trajectory(frame, detection)
            # print(detection, flush=True)
            trajectories[str(frame)].append(new_traj)

    dets = []
    for trajectory in trajectories[str(frame)]:
        dets.append(trajectory)
    if use_metrics:
        metric.log(frame, dets)


    while (frame < sequence.get(cv2.CAP_PROP_FRAME_COUNT) - FRAME_STEP):
    # while (frame <= 1249):

        # Read the next frame
        frame += FRAME_STEP
        sequence.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, next_image = sequence.read()
        # next_image = imresize(next_image, (FRAME_HEIGHT, FRAME_WIDTH, 3))

        for detection, cls in zip(results_1[0]['rois'], results_1[0]['class_ids']):

            # print(class_names[cls])
            if class_names[cls] == 'person':
                cv2.rectangle(image, (detection[1], detection[0]), (detection[3], detection[2]), (255, 255, 0), 3)

        for trajectory in trajectories[str(frame - FRAME_STEP)]:
            coords = trajectory.get_detections()
            cv2.rectangle(image, (coords[1], coords[0]), (coords[3], coords[2]), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for prediction in trajectory.get_predictions():
                cv2.circle(image, (int(prediction[1]), int(prediction[0])), 2, (255, 0, 0), 1)
            area_coords = get_percentile_area(trajectory.get_predictions())
            cv2.rectangle(image, (int(area_coords[1]), int(area_coords[0])), (int(area_coords[3]), int(area_coords[2])), (255, 0, 255), 1)
            cv2.putText(image, str(trajectory.id), (coords[1]+3, coords[0]+15), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)

        trajectories[str(frame)] = []

        # Run the detection on the next frame
        results_1 = model.detect([next_image], verbose=1)

        frame_name = "E:\\Uni\\Level_5_Project\\Tracker\\data\\green\\validation_3_{}.jpg".format(frame-FRAME_STEP)
        cv2.imwrite(frame_name, image)

        candidate_detections = []
        for detection, cls in zip(results_1[0]['rois'], results_1[0]['class_ids']):
            if class_names[cls] == 'person':
                candidate_detections.append(Detection(detection))

        # TODO: Metrics for mask HERE
        # dets = []
        # for detc in candidate_detections:
        #     dets.append(Trajectory(frame,detc.get_coordinates()))
        # if use_metrics:
        #     metric.log(frame, dets)

        # Match detections in frame 0 with detections in frame 1
        previous_trajectories = trajectories[str(frame-FRAME_STEP)].copy()
        for trajectory in previous_trajectories:
            # Compute the area for each preds
            # print("traj {}".format(trajectory.id), flush=True)
            area_coords = get_percentile_area(trajectory.particles)
            det0_coords = trajectory.get_detections(frame-FRAME_STEP)
            potential = []
            # Array to hold the indices of detections_1 that fall within this area
            # Loop through detections in frame 1
            for detection_1 in candidate_detections.copy():

                centroid = get_centroid(detection_1.get_coordinates())
                # print("Candidate Detection centroid: {}".format(centroid), flush=True)

                if area_coords[0] <= centroid[0] <= area_coords[2] and area_coords[1] <= centroid[1] <= \
                        area_coords[3]:
                    # candidate_detections.remove(detection_1)
                    # updated_trajectory = trajectory
                    # trajectories[str(frame - FRAME_STEP)].remove(trajectory)
                    # updated_trajectory.add_detection(frame, detection_1.get_coordinates())
                    # trajectories[str(frame)].append(updated_trajectory)
                    # break
                    potential.append(detection_1)

            if len(potential) > 1:
                similarities = []
                for detection_1 in potential:
                    det1_coords = detection_1.get_coordinates()
                    similarities.append(re_id_m.get_prediction(re_id_model, image[det0_coords[0]:det0_coords[2],
                                                                     det0_coords[1]:det0_coords[3], :],
                                                        next_image[det1_coords[0]:det1_coords[2],
                                                        det1_coords[1]:det1_coords[3], :]))
                    # print("Similarity traj: {}\tsim: {}".format(trajectory.id,similarity))
                    # scipy.misc.imsave(str(det0_coords) + "_1.jpg", image[det0_coords[0]:det0_coords[2],
                    #                                                det0_coords[1]:det0_coords[3], :])
                    # scipy.misc.imsave(str(det1_coords) + "_2.jpg", next_image[det1_coords[0]:det1_coords[2],
                    #                                                det1_coords[1]:det1_coords[3], :])

                ind = similarities.index(max(similarities))
                detection_1 = potential[ind]
                candidate_detections.remove(detection_1)
                updated_trajectory = trajectory
                trajectories[str(frame - FRAME_STEP)].remove(trajectory)
                updated_trajectory.add_detection(frame, detection_1.get_coordinates())
                trajectories[str(frame)].append(updated_trajectory)

            elif len(potential) == 1:
                candidate_detections.remove(potential[0])
                updated_trajectory = trajectory
                trajectories[str(frame - FRAME_STEP)].remove(trajectory)
                updated_trajectory.add_detection(frame, potential[0].get_coordinates())
                trajectories[str(frame)].append(updated_trajectory)


        print("Finished initial matching.", flush=True)
        # Try to match future detections with ones from previous frames
        for detection_1 in candidate_detections.copy():
            det1_coords = detection_1.get_coordinates()
            current_frame = frame - 2 * FRAME_STEP
            matched = False
            while current_frame >= max(frame - 8 * FRAME_STEP, START_FRAME) and not matched:

                previous_trajectories = trajectories[str(current_frame)].copy()
                for trajectory in previous_trajectories:

                    # print("traj {}".format(trajectory.id))
                    area_coords = get_percentile_area(trajectory.particles)
                    # print("traj {} preds: {}".format(trajectory.id,str(trajectory.particles)))
                    centroid = get_centroid(det1_coords)
                    # print("Area coords: {}\tdet_centroid: {}".format(str(area_coords), str(centroid)))
                    if centroid[0] >= area_coords[0] and centroid[0] <= area_coords[2] and centroid[1] >= area_coords[
                        1] and centroid[1] <= area_coords[3]:

                        sequence.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                        ret, previous_detection = sequence.read()
                        # previous_detection = imresize(previous_detection, (FRAME_HEIGHT, FRAME_WIDTH, 3))

                        det0_coords = trajectory.get_detections(frame=current_frame)

                        similarity = re_id_m.get_prediction(re_id_model, previous_detection[det0_coords[0]:det0_coords[2],
                                                                         det0_coords[1]:det0_coords[3], :], next_image[det1_coords[0]:det1_coords[2],
                                                                                                            det1_coords[1]:det1_coords[3], :])
                        print("Similarity traj: {}\tsim: {}".format(trajectory.id,float(similarity)))

                        if similarity >= SIMILARITY_THRESHOLD:
                            candidate_detections.remove(detection_1)

                            trajectories[str(current_frame)].remove(trajectory)
                            trajectory.add_detection(frame, det1_coords)
                            trajectories[str(frame)].append(trajectory)

                            matched = True
                            break

                current_frame -= FRAME_STEP

        # Treat the rest of the detections as new and generate a trajectory object
        for detection_1 in candidate_detections:
            det1_coords = detection_1.get_coordinates()
            # print("Format of det1_coords: {}".format(det1_coords), flush=True)

            # Check if the detection is at the edge of the screen only then create the new trajectory
            centroid = get_centroid(det1_coords)
            print("CENTROID: {}\tFRAME_WIDTH: {}\tFRAME_HEIGHT: {}".format(centroid,FRAME_WIDTH,FRAME_HEIGHT))
            if (centroid[0] <= FRAME_WIDTH * FRAME_BOUNDARY or centroid[
                0] >= FRAME_WIDTH - FRAME_WIDTH * FRAME_BOUNDARY) or \
                    (centroid[1] <= FRAME_HEIGHT * FRAME_BOUNDARY or centroid[
                        1] >= FRAME_HEIGHT - FRAME_HEIGHT * FRAME_BOUNDARY):
                trajectories[str(frame)].append(Trajectory(frame, det1_coords))

        # cv2.imshow('output', image)
        # cv2.waitKey(delay=1)

        image = next_image

        # TODO: Metrics for the whole system
        if use_metrics:
            dets = []
            for trajectory in trajectories[str(frame)]:
                dets.append(trajectory)
            if use_metrics:
                metric.log(frame, dets)

        print(trajectories, flush=True)
        # cv2.destroyAllWindows()

    with open("E:\\Uni\\Level_5_Project\\Tracker\\data\\green\\validation_3_no_sim.pickle", 'wb') as output_file:
        pickle.dump(trajectories, output_file)
    if use_metrics:
        with open("e:/Uni/Level_5_Project/Tracker/17-04-18_while_system_no_vel.txt",'w') as out:
            out.write(metric.print_stats())



def get_centroid(detection):
    return (detection[0] + ((detection[2] - detection[0]) / 2), detection[1] + ((
                                                                                    detection[3] - detection[1]) / 2))


def get_hard_area(predictions):
    min_y = FRAME_WIDTH
    max_y = -FRAME_WIDTH

    min_x = FRAME_HEIGHT
    max_x = -FRAME_HEIGHT

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
    # y_padding = (max_y - min_y) * 0.4
    # x_padding = (max_x - min_x) * 0.4

    y_padding = 0
    x_padding = 0

    min_x, max_x = min_x - x_padding, max_x + x_padding
    min_y, max_y = min_y - y_padding, max_y + y_padding
    return (min_x, min_y, max_x, max_y)

def get_percentile_area(predictions):
    preds = np.array(predictions)
    mean = preds.mean(axis=0)
    std = preds.std(axis=0)

    buffer = 2
    return (mean[0] - (std[0] * buffer),mean[1] - std[1]*buffer, mean[0] + std[0]*buffer, mean[1] + std[1]*buffer)


if __name__ == "__main__":

    # main(*sys.argv[1:], reid_weights="E:\\Uni\\Level_5_Project\\Tracker\\re_id\\cuhk03\\weights\\weights_on_cuhk03_0_0.pickle",)
    main("E:\\Uni\\Level_5_Project\\Tracker\\data\\green\\validation_3.avi",
         reid_weights="E:\\Uni\\Level_5_Project\\Tracker\\re_id\\cuhk03\\weights\\weights_on_cuhk03_0_0.pickle",
         gt="e:/Uni/Level_5_Project/Tracker/data/green/validation_3/allD_id.txt",
         use_metrics=True)
