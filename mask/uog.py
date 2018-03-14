import numpy as np
import os
import scipy.misc
import mask.utils as utils
import matplotlib.pyplot as plt
from mask.config import Config

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_hda.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")
DEFAULT_DATASET_YEAR = "2015"


############################################################
#  Configurations
############################################################


class UOGConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "uog"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # UOG has 1 class

    # IMAGE_SHAPE = (448,640,3)


############################################################
#  Dataset
############################################################

class UOGDataset(utils.Dataset):

    def load_uog(self, sequence, dataset_dir, height=338, width=600):

        # Add classes
        self.add_class("uog", 1, "person")
        allDetections = open("{}\\{}\\allD.txt".format(dataset_dir, sequence),'r')
        for line in allDetections:
            line = line.split()
            detection_frame = line[0]
            s = sequence.split("\\")[0]
            # Add images
            img_name = "o{}_{}.jpg".format(s, detection_frame)
            mask_name = "{}\\{}\\masks\\m{}_{}.jpg".format(dataset_dir, sequence, s, detection_frame)
            img_id = "{}_{}".format(s, img_name)
            detections = [(float(line[1]), float(line[2]), float(line[1]) + float(line[3]), float(line[2]) + float(line[4]))]
            self.add_image("uog", image_id=img_id,
                           path="{}\\{}\\images\\{}".format(dataset_dir, sequence, img_name),
                           width=width, height=height, detections=detections, mask=mask_name)


        allDetections.close()

    def add_image(self, source, image_id, path, **kwargs):

        if self.image_info and self.image_info[-1]['id'] == image_id:
            image_info = self.image_info.pop()
            detections = image_info['detections']
            dets = kwargs['detections']
            detections = detections + dets
            image_info['detections'] = detections
        else:
            image_info = {
                "id": image_id,
                "source": source,
                "path": path,
            }
            image_info.update(kwargs)
        self.image_info.append(image_info)


    def load_mask(self, image_id):
        """Generate instance masks for the bounding boxes given image ID.
        """
        info = self.image_info[image_id]
        detections = info['detections']
        mask = scipy.misc.imread(info['mask'], mode='L')
        mask = mask / 255
        print(np.amax(mask))
        # plt.imshow(mask)
        # plt.show()
        mask = np.reshape(mask,(mask.shape[0],mask.shape[1],1))
        count = len(detections)
        final_mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (x_0, y_0, x_1, y_1) in enumerate(info['detections']):
            final_mask[int(y_0):int(y_1), int(x_0):int(x_1), i:i + 1] = 1
            final_mask[:,:,i:i+1] = final_mask[:,:,i:i+1] * mask

        class_ids = np.array([self.class_names.index('person') for d in detections])

        return final_mask, class_ids.astype(np.int32)
