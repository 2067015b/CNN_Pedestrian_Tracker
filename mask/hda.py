import numpy as np
import os
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


class HDAConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """
    # Give the configuration a recognizable name
    NAME = "hda"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # HDA has 1 class

    # IMAGE_SHAPE = (448,640,3)


############################################################
#  Dataset
############################################################

class HDADataset(utils.Dataset):
    def load_coco(self, dataset_dir, subset, year=DEFAULT_DATASET_YEAR, class_ids=None,
                  class_map=None, return_coco=False, auto_download=False):
        """Load a subset of the COCO dataset.
        dataset_dir: The root directory of the COCO dataset.
        subset: What to load (train, val, minival, valminusminival)
        year: What dataset year to load (2014, 2017) as a string, not an integer
        class_ids: If provided, only loads images that have the given classes.
        class_map: TODO: Not implemented yet. Supports maping classes from
            different datasets to the same class ID.
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)

        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if subset == "minival" or subset == "valminusminival":
            subset = "val"
        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

        # Load all classes or a subset?
        if not class_ids:
            # All classes
            class_ids = sorted(coco.getCatIds())

        # All images or a subset?
        if class_ids:
            image_ids = []
            for id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[id])))
            # Remove duplicates
            image_ids = list(set(image_ids))
        else:
            # All images
            image_ids = list(coco.imgs.keys())

        # Add classes
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        # Add images
        for i in image_ids:
            self.add_image(
                "coco", image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                width=coco.imgs[i]["width"],
                height=coco.imgs[i]["height"],
                annotations=coco.loadAnns(coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_coco:
            return coco

    def load_shapes(self, count, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.add_class("shapes", 1, "square")
        self.add_class("shapes", 2, "circle")
        self.add_class("shapes", 3, "triangle")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for i in range(count):
            bg_color, shapes = self.random_image(height, width)
            self.add_image("shapes", image_id=i, path=None,
                           width=width, height=height,
                           bg_color=bg_color, shapes=shapes)

    def load_hda(self, camera, count, dataset_dir, height=480, width=640):

        # Add classes
        self.add_class("hda", 1, "person")
        current_frame = 0
        allD = open("{}\\hda_detections\\GtAnnotationsAll\\camera{}\\Detections\\allD.txt".format(dataset_dir, camera),
                    'r')
        for line in allD:
            line = line.split(',')
            detection_frame = int(line[1])
            while detection_frame > current_frame and current_frame <= count:
                # Add images
                img_name = "I{0:0>5}.jpg".format(current_frame)
                img_id = "{}_{}".format(camera,img_name)
                self.add_image("hda", image_id=img_id,
                               path="{}\\hda_image_sequences_matlab\\camera{}\\{}".format(dataset_dir, camera, img_name),
                               width=width, height=height, detections=[])
                current_frame += 1

            if current_frame == detection_frame:
                detections = [(float(line[2]), float(line[3]), float(line[2]) + float(line[4]), float(line[3]) + float(line[5]))]
                img_name = "I{0:0>5}.jpg".format(current_frame)
                img_id = "{}_{}".format(camera,img_name)
                self.add_image("hda", image_id=img_id,
                               path="{}\\hda_image_sequences_matlab\\camera{}\\{}".format(dataset_dir, camera, img_name),
                               width=width, height=height, detections=detections, catIds=class_ids)

        allD.close()

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

    # def load_mask(self, image_id):
    #     """Generate instance masks for shapes of the given image ID.
    #     """
    #     info = self.image_info[image_id]
    #     shapes = info['shapes']
    #     count = len(shapes)
    #     mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
    #     for i, (shape, _, dims) in enumerate(info['shapes']):
    #         mask[:, :, i:i + 1] = self.draw_shape(mask[:, :, i:i + 1].copy(),
    #                                               shape, dims, 1)
    #     # Handle occlusions
    #     occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
    #     for i in range(count - 2, -1, -1):
    #         mask[:, :, i] = mask[:, :, i] * occlusion
    #         occlusion = np.logical_and(
    #             occlusion, np.logical_not(mask[:, :, i]))
    #     # Map class names to class IDs.
    #     class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
    #     return mask, class_ids.astype(np.int32)

    def load_mask(self, image_id):
        """Generate instance masks for the bounding boxes given image ID.
        """
        info = self.image_info[image_id]
        detections = info['detections']
        count = len(detections)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (x_0, y_0, x_1, y_1) in enumerate(info['detections']):
            mask[int(y_0):int(y_1), int(x_0):int(x_1), i:i + 1] = 1

        class_ids = np.array([self.class_names.index('person') for d in detections])

        return mask, class_ids.astype(np.int32)
