import sys
import os
import random
import numpy as np
import re
from PIL import Image
import matplotlib.pyplot as plt
from Augmentor import Pipeline
from scipy import misc


def flip(input_dir, output_dir):
    resolution = (600, 338)

    for filename in os.listdir(input_dir + "/images/"):
        img = misc.imread("{}/images/{}".format(input_dir, filename))

        filename = filename.split('_')
        im_number = float(filename[-1].split('.')[0]) + 3000

        img = np.flip(img, 1)
        misc.imsave("{}/images/ovalidation_1_{}.jpg".format(output_dir, im_number), img)

    for filename in os.listdir(input_dir + "/masks/"):
        mask = misc.imread("{}/masks/{}".format(input_dir, filename))

        filename = filename.split('_')
        im_number = float(filename[-1].split('.')[0]) + 3000

        mask = np.flip(mask, 1)
        misc.imsave("{}/masks/mvalidation_1_{}.jpg".format(output_dir, im_number), mask)

    allDetections = open("{}\\allD.txt".format(input_dir), 'r')
    new_detections = open("{}\\new_allD.txt".format(output_dir), 'w')

    for line in allDetections:
        new_detections.write(line)
        line = line.split()
        new_x = resolution[0] - int(line[1]) - int(line[3])
        new_detection = "{} {} {} {} {}\n".format(float(line[0]) + 3000, new_x, line[2], line[3], line[4])
        new_detections.write(new_detection)

    allDetections.close()
    new_detections.close()


def occlude_boxes(input_dir, output_dir):
    all_detections = open("{}\\allD_id.txt".format(input_dir), 'r')

    for line in all_detections:
        line = line.split()
        try:
            img = misc.imread("{}/temp/validation_3_{}.jpg".format(input_dir, line[0]))
        except:
            img = misc.imread("{}/images/validation_3_{}.jpg".format(input_dir, line[0]))
        x1, y1, x2, y2 = int(line[1]), int(line[2]), int(line[1]) + int(line[3]), int(line[2]) + int(line[4])

        rect = random_erasing(img[y1:y2 + 1, x1:x2 + 1, :])

        img[y1:y2 + 1, x1:x2 + 1, :] = rect[:, :, :]

        misc.imsave("{}/temp/validation_3_{}.jpg".format(input_dir, line[0]), img)


        # misc.imsave("{}/temp/ovalidation_1_{}_{}.jpg".format(input_dir, line[0], i), img)
    all_detections.close()


def get_frame_from_filename(filename):
    # ovalidation_1_1251.0.jpg
    parts = filename.replace('_', '.').split('.')
    return int(parts[2])


def overlaps(detection, occlusion):
    x_1d = detection[0]
    x_2d = detection[0] + detection[2]
    y_1d = detection[1]
    y_2d = detection[1] + detection[3]

    x_1o = occlusion[0]
    x_2o = occlusion[0] + occlusion[2]
    y_1o = occlusion[1]
    y_2o = occlusion[1] + occlusion[3]

    if (x_1d >= x_2o) or (x_2d <= x_1o) or (y_1d >= y_2o) or (y_2d <= y_1o):
        return detection
    elif x_1o < x_1d:
        if y_1o < y_1d:
            if (x_2o - x_1d) / detection[2] < (y_2o - y_1d) / detection[3]:
                x_1d = x_2o
            else:
                y_1d = y_2o
        else:
            # case 4
            if (x_2o - x_1d) / detection[2] < (y_2d - y_1o) / detection[3]:
                x_2d = x_1o
            else:
                y_2d = y_1o
    elif y_1o < y_1d:
        # case 2
        if (x_2d - x_1o) / detection[2] < (y_2o - y_1d) / detection[3]:
            x_1d = x_2o
        else:
            y_2d = y_1o
    else:
        # case 3
        if (x_2d - x_1o) / detection[2] < (y_2d - y_1o) / detection[3]:
            x_2d = x_1o
        else:
            y_2d = y_1o

    return [x_1d, y_1d, x_2d - x_1d, y_2d - y_1d, detection[-1]]


def occlude_all(input_dir, output_dir, annotations=None):
    if annotations:
        gt = {}
        with open(annotations, 'r') as gt_file:
            for line in gt_file:
                line = line.split()
                current_frame = int(float(line[0]))
                current_det = []
                for coordinate in line[1:-1]:
                    current_det.append(int(coordinate))
                current_det.append(line[-1])
                if len(current_det) < 5:
                    print(current_det)
                dets = gt.get(current_frame, [])
                dets.append(current_det)
                gt[current_frame] = dets

    coords = None
    for filename in os.listdir(input_dir):
        img = misc.imread(input_dir + filename)
        img, coords = erasing(img, coords=coords)
        misc.imsave(output_dir + filename, img)
        if annotations:
            frame_no = get_frame_from_filename(filename)
            dets = []
            for detection in gt[frame_no]:
                dets.append(overlaps(detection, coords))
            gt[frame_no] = dets

    if annotations:
        with open(output_dir + "allD.txt", 'w') as out:
            for key in gt.keys():
                for detection in gt[key]:
                    print(detection)
                    out.write("{}.0 {} {} {} {} {}\n".format(key, *detection))




                    # misc.imsave("{}/temp/ovalidation_1_{}_{}.jpg".format(input_dir, line[0], i), img)

                    # i = 0
                    # for line in all_detections:
                    #     i += 1
                    #     line = line.split()
                    #     try:
                    #         orig = misc.imread("{}/temp/ovalidation_1_{}.jpg".format(input_dir, line[0]))
                    #     except:
                    #         orig = misc.imread("{}/images/ovalidation_1_{}.jpg".format(input_dir, line[0]))
                    #
                    #     rect = misc.imread("{}/temp/ovalidation_1_{}_{}.jpg".format(input_dir, line[0], i))
                    #     x1, y1, x2, y2 = int(line[1]), int(line[2]), int(line[1]) + int(line[3]), int(line[2]) + int(line[4])
                    #     orig[y1:y2+1, x1:x2+1, :] = rect[:,:,:]
                    #     misc.imsave("{}/temp/ovalidation_1_{}.jpg".format(input_dir, line[0]), orig)


def random_erasing(image, rectangle_area=0.4):
    """
    Adds a random noise rectangle to a random area of the passed image,
    returning the original image with this rectangle superimposed.
    :param image: The image to add a random noise rectangle to.
    :type image: PIL.Image
    :return: The image with the superimposed random rectangle as type
     image PIL.Image
    """

    w, h = image.shape[0], image.shape[1]

    w_occlusion_max = int(w * rectangle_area)
    h_occlusion_max = int(h * rectangle_area)

    w_occlusion_min = int(w * 0.2)
    h_occlusion_min = int(h * 0.2)

    w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
    h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)

    if image.shape[2] == 1:
        rectangle = np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255)
    else:
        rectangle = np.uint8(np.random.rand(w_occlusion, h_occlusion, image.shape[2]) * 255)

    random_position_x = random.randint(0, w - int(w_occlusion))
    random_position_y = random.randint(0, h - int(h_occlusion))

    image[random_position_x:random_position_x + int(w_occlusion),
    random_position_y:random_position_y + int(h_occlusion), :] = rectangle
    return image, [random_position_x, random_position_y, w_occlusion, h_occlusion]


def erasing(image, coords=None, rectangle_area=0.4):
    """
    Adds a random noise rectangle to a random area of the passed image,
    returning the original image with this rectangle superimposed.
    :param image: The image to add a random noise rectangle to.
    :type image: PIL.Image
    :return: The image with the superimposed random rectangle as type
     image PIL.Image
    """

    w, h = image.shape[0], image.shape[1]

    w_occlusion_max = int(w * rectangle_area)
    h_occlusion_max = int(h * rectangle_area)

    w_occlusion_min = int(w * 0.2)
    h_occlusion_min = int(h * 0.2)

    if not coords:
        w_occlusion = random.randint(w_occlusion_min, w_occlusion_max)
        h_occlusion = random.randint(h_occlusion_min, h_occlusion_max)
    else:
        w_occlusion = coords[2]
        h_occlusion = coords[3]

    if image.shape[2] == 1:
        rectangle = np.uint8(np.random.rand(w_occlusion, h_occlusion) * 255)
    else:
        rectangle = np.uint8(np.random.rand(w_occlusion, h_occlusion, image.shape[2]) * 255)

    random_position_x = random.randint(0, w - int(w_occlusion))
    random_position_y = random.randint(0, h - int(h_occlusion))

    if coords:
        if coords[0] < 0.5 * w:
            random_position_x = random.randint(coords[0] - 10, min(coords[0] + 11, w - int(w_occlusion)))
        else:
            random_position_x = random.randint(coords[0] - 11, min(coords[0] + 10, w - int(w_occlusion)))
        if coords[1] < 0.5 * h:
            random_position_y = random.randint(coords[1] - 10, min(coords[1] + 11, h - int(h_occlusion)))
        else:
            random_position_y = random.randint(coords[1] - 11, min(coords[1] + 10, h - int(h_occlusion)))

    image[random_position_x:random_position_x + int(w_occlusion),
    random_position_y:random_position_y + int(h_occlusion), :] = rectangle
    return image, [random_position_x, random_position_y, w_occlusion, h_occlusion]


def remove_negative():
    with open("data/green/validation_3/occluded/consistent/allD.txt", 'r') as inp:
        with open("data/green/validation_3/occluded/consistent/allD_c.txt", 'w') as out:
            for line in inp:
                if int(line.split()[-2]) > 0 and int(line.split()[-3]) > 0:
                    out.write(line)
                else:
                    print(line)


if __name__ == "__main__":
    # occlude_boxes(*sys.argv[1:])
    occlude_all("data/green/validation_3/images/", "data/green/validation_3/occluded/consistent/",
                "data/green/validation_3/allD_id.txt")
    remove_negative()
