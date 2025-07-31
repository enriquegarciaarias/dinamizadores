from sources.common.common import logger, processControl, log_

import torch
import os
import shutil
import joblib
import numpy as np
import json

def save_clusters(centroids, cluster_labels, image_classes, pca):
    try:
        # Guardar centroides
        filePath = os.path.join(processControl.env['models'], 'centroids.npy')
        np.save(filePath, centroids)
        # Guardar etiquetas correspondientes a cada cluster
        filePath = os.path.join(processControl.env['models'], 'labels.npy')
        np.save(filePath, np.array(image_classes))
        # Guardar los labels predichos por el modelo de clustering para futura asignación
        filePath = os.path.join(processControl.env['models'], 'cluster_labels.npy')
        np.save(filePath, cluster_labels)
        filepath = os.path.join(processControl.env['models'], "pca_transform.pkl")
        joblib.dump(pca, filepath)
    except Exception as e:
        raise e

def load_clusters():
    filePath = os.path.join(processControl.env['models'], 'centroids.npy')
    centroids = np.load(filePath)
    filePath = os.path.join(processControl.env['models'], 'labels.npy')
    labels = np.load(filePath)
    filePath = os.path.join(processControl.env['models'], 'cluster_labels.npy')
    cluster_labels = np.load(filePath)
    filePath = os.path.join(processControl.env['models'], "pca_transform.pkl")
    pca = joblib.load(filePath)
    return centroids, labels, cluster_labels, pca

def saveModel(model, type):
    """
    Save the model to a specified file based on its type.

    This function saves the trained model to a file depending on the specified type. It supports saving LightGBM models
    as `.pkl` files and PyTorch models as `.pth` files. If an error occurs during the saving process, it raises an exception.

    :param model: The trained model to be saved.
    :type model: object
    :param type: The type of the model, which determines the file format to be used.
    :type type: str

    :return: The path where the model was saved.
    :rtype: str

    :raises Exception: If an error occurs during the model saving process.
    """
    try:
        if type == "lightgbm":
            modelPath = os.path.join(processControl.env['models'], "lightgbm_model.pkl")
            joblib.dump(model, modelPath)

        if type == "features":
            modelPath = os.path.join(processControl.env['outputPath'], "features.pth")
            torch.save(model, modelPath)

    except Exception as e:
        raise Exception(f"Couldn't save model: {e}")

    log_("info", logger, f"Model type: {type} saved to {modelPath}")
    return modelPath


def loadModelOpenClip(modelName, pretrainedDataset):
    import open_clip
    try:
        model, preprocess, _ = open_clip.create_model_and_transforms(modelName,pretrainedDataset)
        model.eval()
    except Exception as e:
        raise e

    return model, preprocess


def writeFilesCategories(clusteredImages, model):
    """
    Organize images into directories based on their cluster labels.

    This function takes a dictionary of clustered images, where each key is a cluster label and
    the corresponding value is a list of image names. It creates directories named by cluster labels and
    moves the images into the appropriate directories.

    :param clustered_images: A dictionary mapping cluster labels to lists of image names belonging to those clusters.
    :type clustered_images: dict

    :return: None
    :rtype: None
    """
    dirModelPath = os.path.join(processControl.env['outputPath'], model)
    if not os.path.exists(dirModelPath):
        os.makedirs(dirModelPath)
    for index, image_info in enumerate(clusteredImages):

        dirCategory = os.path.join(dirModelPath, f"category_{image_info['category']}")
        if not os.path.exists(dirCategory):
            os.makedirs(dirCategory)
        shutil.copy(image_info['path'], os.path.join(dirCategory, image_info['name']))

    log_("info", logger, f"Images organized into directories.")


def structureFiles(clustered_images, model):
    """
    Organize images into directories based on their cluster labels.

    This function takes a dictionary of clustered images, where each key is a cluster label and
    the corresponding value is a list of image names. It creates directories named by cluster labels and
    moves the images into the appropriate directories.

    :param clustered_images: A dictionary mapping cluster labels to lists of image names belonging to those clusters.
    :type clustered_images: dict

    :return: None
    :rtype: None
    """
    dirModelPath = os.path.join(processControl.env['outputPath'], model)
    if not os.path.exists(dirModelPath):
        os.makedirs(dirModelPath)
    for index, images in clustered_images.items():
        # Create directory name
        dir_name = os.path.join(dirModelPath, f"images_{index}")

        # Create the directory if it doesn't exist
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

        # Move images to the directory
        for image in images:
            # Assuming images are in the current working directory
            imgSource = os.path.join(processControl.env['inputPath'], image)
            if os.path.exists(imgSource):
                shutil.copy(imgSource, os.path.join(dir_name, image))
            else:
                log_("error", logger, f"Image {image} not found.")

    log_("info", logger, f"Images organized into directories.")

def readResults(stage):
    file_path = os.path.join(processControl.env['outputPath'], f"results_{stage}.json")
    if os.path.exists(file_path):
        log_("info", logger, f"The file '{file_path}' exists.")
        try:
            # Open and read the JSON file
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)  # Parse the JSON content

        except Exception as e:
            log_("error", logger, f"An unexpected error occurred while reading the file: {e}")
            return False
    else:
        return False
    return data


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, set):
        return list(obj)
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    else:
        raise TypeError(f"Type {type(obj)} not serializable")


def writeResultsData(data, stage):
    try:
        resultsPath = os.path.join(processControl.env['outputPath'], f"results_{stage}.json")
        with open(resultsPath, 'w') as json_file:
            json.dump(data, json_file, indent=4, ensure_ascii=False, default=convert_to_serializable)
    except Exception as e:
        raise Exception(f"Couldn't save results: {e}")

    return True
