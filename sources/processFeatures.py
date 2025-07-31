from sources.common.common import logger, processControl, log_
from sources.dataManager import saveModel, save_clusters, load_clusters, loadModelOpenClip, structureFiles
from sources.common.utils import buildImageProcess

import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from transformers import BertTokenizer, BertModel

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# VIT ViT (Vision Transformer) trained on LAION-2B performs better for your archaeology images compared to CLIP

def extractFeatures(imagesList, mode="VIT"):
    # Cargar la imagen
    image_features = {}
    device = processControl.defaults['device']
    try:
        if mode == "clip":
            import open_clip
            from transformers import CLIPProcessor, CLIPModel
            # Cargar el modelo y el procesador de CLIP
            model_name = "openai/clip-vit-base-patch32"
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name).to(device)
            for imagePro in tqdm(imagesList, desc="Extracting features"):
                image = Image.open(imagePro['imagePath']).convert("RGB")
                inputs = processor(images=image, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    features = model.get_image_features(**inputs).squeeze(0).cpu()
                image_features[imagePro['name']] = features


        elif mode == "dino":
            from transformers import Dinov2Model, Dinov2ImageProcessor
            # Cargar el modelo y el procesador de DINOv2
            model_name = "facebook/dinov2-base"
            processor = Dinov2ImageProcessor.from_pretrained(model_name)
            model = Dinov2Model.from_pretrained(model_name)
            for imagePro in tqdm(imagesList, desc="Extracting features"):
                image = Image.open(imagePro['imagePath']).convert("RGB")
                inputs = processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    features = model(**inputs).last_hidden_state.mean(dim=1)
                image_features[imagePro['name']] = features

        elif mode == "VIT":
            model, preprocess = loadModelOpenClip(
                processControl.models['VIT']['modelName'],
                processControl.models['VIT']['pretrainedDataset']
            )
            model.to(device)
            for imagePro in tqdm(imagesList, desc="Extracting features"):
                image = preprocess(Image.open(imagePro['imagePath']).convert("RGB")).unsqueeze(0).to(device)
                with torch.no_grad():
                    features = model.encode_image(image).squeeze(0).cpu() # Move to CPU for storage
                image_features[imagePro['name']] = features
    except Exception as e:
        raise Exception(f"Error extracting features {e}")

    return image_features


def extractFeaturesForInference(imageFolder):
    """
    Extract features from images in a specified folder for inference.

    This function extracts features from images stored in a folder, typically used for inference. The extracted features
    are returned as a numpy array, which can then be processed or used for prediction.

    :param imageFolder: Path to the folder containing images for which features are to be extracted.
    :type imageFolder: str

    :return: A numpy array containing the extracted features for each image in the folder.
    :rtype: numpy.ndarray
    """
    imagesList = buildImageProcess(imageFolder)
    features = extractFeatures(imagesList, processControl.args.featuresmodel)
    image_features = []
    for feature in features.values():
        image_features.append(feature)
    return np.array(image_features)


def assign_to_cluster(new_image_features):
    # Cargar centroides y labels previamente guardados
    centroids, labels, cluster_labels, pca = load_clusters()

    # Reducir las características de la nueva imagen (usando PCA)
    new_image_reduced = pca.transform(new_image_features.reshape(1, -1))

    # Calcular la similaridad con los centroides (por ejemplo, utilizando cosine similarity)
    similarities = cosine_similarity(new_image_reduced, centroids)

    # Encontrar el índice del centroide más cercano
    closest_cluster_idx = np.argmax(similarities)

    # Asignar la nueva imagen a la categoría correspondiente
    assigned_label = labels[closest_cluster_idx]  # Etiqueta asociada al cluster
    return assigned_label, closest_cluster_idx


def optimizeDimensions(image_features):
    """
    Optimize the dimensionality of image features using PCA.

    This function applies Principal Component Analysis (PCA) to reduce the dimensionality
    of the extracted image features while retaining the most important variance. It computes
    the cumulative explained variance and plots it to help select the optimal number of PCA components
    that capture a desired level of variance (e.g., 95%).

    :param image_features: A dictionary mapping image names to their corresponding feature tensors.
    :type image_features: dict

    :return: A dictionary mapping image names to their corresponding reduced feature arrays.
    :rtype: dict

    :note: The number of PCA components is dynamically chosen to capture at least 95% of the explained variance.
    """
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    feature_matrix = np.array([tensor.numpy() for tensor in image_features.values()])
    image_names = list(image_features.keys())
    #feature_matrix = feature_matrix.squeeze(axis=1)
    # Initialize PCA without specifying n_components to analyze variance
    pca = PCA()
    pca.fit(feature_matrix)

    # Compute cumulative explained variance
    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot variance to choose an optimal number of components
    plt.plot(range(1, len(explained_variance) + 1), explained_variance)
    plt.xlabel('Number of PCA Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance by PCA Components')
    plt.grid()
    plt.show()

    # Choose n_components dynamically, e.g., for 95% explained variance
    optimal_components = np.argmax(explained_variance >= 0.95) + 1
    log_("info", logger, f"Optimal number of PCA components: {optimal_components}")
    # Apply PCA with the optimal number of components
    pca = PCA(n_components=optimal_components)
    reduced_features = pca.fit_transform(feature_matrix)
    # ✅ Convert reduced feature array back into a dictionary
    reduced_feature_dict = {image_names[i]: reduced_features[i] for i in range(len(image_names))}

    return reduced_feature_dict


def clusterImages(featuresFile=None, image_features=None):
    """
    Perform clustering on image features to group images based on similarity.

    This function loads precomputed image features from a specified file, applies PCA for dimensionality reduction,
    and then clusters the images using KMeans. The images are grouped into clusters based on their feature vectors.

    :param featuresFile: Path to the file containing saved image features.
    :type featuresFile: str

    :return: A tuple containing:
        - clustered_images (dict): A dictionary mapping cluster labels to lists of image names.
        - centroids (numpy.ndarray): The cluster centers after the KMeans clustering.
    :rtype: tuple
    """
    if featuresFile:
        # Load saved features
        # weights_only=True se puede incluir para evitar warnings dado que solo queremos cargar los pesos del modelo
        # image_features dict 'imagename':tensor(1,512)
        image_features = torch.load(featuresFile, weights_only=True)

        # Convert feature tensors to a matrix for clustering
        #feature_matrix ndarray(69,1,512)
        feature_matrix = torch.stack(list(image_features.values())).numpy()

    else:
        #image_features "img":Tensor (768,) en VIT
        feature_matrix = np.array([tensor.numpy() for tensor in image_features.values()])

    # Dimensionality reduction (optional)
    pca = PCA(n_components=processControl.defaults['features'])  # Reduce dimensions
    reduced_features = pca.fit_transform(feature_matrix)
    # Clustering
    num_clusters = len(processControl.defaults['imageClasses'])
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(reduced_features)  # Train clustering
    cluster_labels = kmeans.predict(reduced_features)

    centroids = kmeans.cluster_centers_

    # Assign images to clusters
    clustered_images = {i: [] for i in range(num_clusters)}
    for idx, image_name in enumerate(image_features.keys()):
        clustered_images[cluster_labels[idx]].append(image_name)

    # Print clusters
    for cluster, images in clustered_images.items():
        log_("info", logger, f"Cluster: {cluster} clustering {len(images)} images")

    return clustered_images, centroids, cluster_labels, pca


def processFeatures():
    """
    Extract, optimize, cluster, and organize image features.

    This function performs the following steps:
    1. Extracts features from images in the specified input directory.
    2. Optimizes the dimensionality of the extracted features using PCA.
    3. Saves the extracted features to a file.
    4. Clusters the images based on their features.
    5. Organizes the images into directories based on their cluster assignments.
    6. Builds a mapping of images to their respective cluster labels.

    :return: A tuple containing:
        - featuresFile (str): The path to the file containing the extracted image features.
        - imagesLabels (dict): A dictionary mapping image names to their corresponding cluster labels.
    :rtype: tuple
    """
    imageFeatures = {}
    try:
        imagesList = buildImageProcess(processControl.env['inputPath'])
        imageFeatures = extractFeatures(imagesList, processControl.args.featuresmodel)

        # Step 2: Optimize the dimensionality of the extracted features using PCA
        imageFeatures2 = optimizeDimensions(imageFeatures)

        # Step 3: Save the extracted features to a file
        featuresFile = saveModel(imageFeatures, "features")

        # Step 4: Cluster the images based on their features
        clusteredImages, centroids, cluster_labels, pca = clusterImages(None, imageFeatures)
        save_clusters(centroids, cluster_labels, processControl.defaults['imageClasses'], pca)

        # Step 5: Organize the images into directories based on their clusters
        structureFiles(clusteredImages, processControl.args.featuresmodel)

    except Exception as ex:
        log_("exception", logger, ex)
        return None

    # Return the path to the features file and the image-to-label mapping
    return True
