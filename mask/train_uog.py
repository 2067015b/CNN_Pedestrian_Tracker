import os
import sys
import model as modellib
from uog import UOGDataset, UOGConfig


def train(args):

    # Root directory of the project
    ROOT_DIR = os.getcwd()
    ROOT_DIR = ROOT_DIR + "/mask_r-cnn"

    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

    DATASET_DIR = args[0]

    config = UOGConfig()
    config.display()
    # Create model in training mode
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=MODEL_DIR)

    # Training dataset
    dataset_train = UOGDataset()
    dataset_train.load_uog('validation_1', DATASET_DIR)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = UOGDataset()
    dataset_val.load_uog('validation_1\\val', DATASET_DIR)
    dataset_val.prepare()


    # Which weights to start with?
    init_with = "coco"  # imagenet, coco, or last

    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last()[1], by_name=True)

    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')

    model.keras_model.save_weights(os.path.join(ROOT_DIR, 'weights/uog.h5'))

if __name__ == "__main__":
    train(sys.argv[1:])