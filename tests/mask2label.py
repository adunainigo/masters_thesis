import cv2
import numpy as np
from scipy import ndimage
from skimage import measure, morphology, feature
import matplotlib.pyplot as plt
import os
import json
import re

#DEBO MODIFICAR ESTE FICHERO CON LOS SIGUIENTES CAMBIOS

# Define hyperparameters
MASKDIR = "./predicted_images/pred_19.jpg"
PRED_MASK_DIR = './predicted_images'
LABELS_DIR ='./data/labels'
LABELS_GT_DIR= "./data/labels_gt"
def preprocess_image(mask_path):
    """
    Given a mask path, returns a dictionary with the features of each piece in 
    the image. 
    """
    # Read the image mask
    mask = cv2.imread(mask_path, 0)
    
    # Apply Gaussian blur to reduce noise while preserving edges
    blur = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Threshold the image to ensure only the pieces are in white
    _, thresh = cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
    
    # Find all contours on the thresholded image
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter out very small contours that are likely noise
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 100]
    
    # If no contours remain after filtering, it's an error
    if not contours:
        raise ValueError("No contours found after noise reduction.")
    
    # Initialize list to hold features of each piece
    pieces_features = []
    
    # Process each contour to extract features
    for piece_contour in contours:
        # Create a mask of the piece
        piece_mask = np.zeros_like(mask)
        cv2.drawContours(piece_mask, [piece_contour], -1, (255), thickness=cv2.FILLED)
        
        # Calculate aspect ratio (boundingRect)
        x, y, w, h = cv2.boundingRect(piece_contour)
        aspect_ratio = float(w) / h

        # Eccentricity (fitEllipse) if the contour has enough points
        if piece_contour.shape[0] >= 5:
            (x, y), (MA, ma), angle = cv2.fitEllipse(piece_contour)
            eccentricity = np.sqrt(1 - (MA / ma) ** 2)
        else:
            eccentricity = None

        # Area and perimeter
        area = cv2.contourArea(piece_contour)
        perimeter = cv2.arcLength(piece_contour, True)

        # Centroid and orientation (moments)
        M = cv2.moments(piece_contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        centroid = (cx, cy)

        # Orientation (moments) if possible to calculate
        orientation = 0.5 * np.arctan((2 * M['mu11']) / (M['mu20'] - M['mu02'])) if M['mu20'] != M['mu02'] else None

        # Skeletonization (morphology.skeletonize)
        skeleton = morphology.skeletonize(piece_mask == 255)

        # Convexity (convexHull)
        hull = cv2.convexHull(piece_contour)
        convexity = cv2.isContourConvex(hull)

        # Compile features into a dictionary
        features = {
            'aspect_ratio': aspect_ratio,
            'eccentricity': eccentricity,
            'area': area,
            'perimeter': perimeter,
            'centroid': centroid,
            'orientation': orientation,
            'convexity': convexity,
        }
        
        # Add features of the current piece to the list
        pieces_features.append(features)

    return pieces_features

def predict_labels(PRED_MASK_DIR, SAVING_FOLDER):
    """
    Loads data, predicts the labels and saves it in a folder label_i.txt
    """
    mask_files = os.listdir(PRED_MASK_DIR)
    for mask_file in mask_files:
        match = re.match(r'pred_mask_(\d+)\.jpg', mask_file)
        if match:
            i = int(match.group(1))
            mask_path = os.path.join(PRED_MASK_DIR, mask_file)
            pieces_features = preprocess_image(mask_path)
    
            save_path = os.path.join(SAVING_FOLDER, f'label_{i}.txt')
            
            with open(save_path, 'w') as file:
                json.dump(pieces_features, file)


def normalise_predicted_data(directory):
    """
    Loads data from JSON files, normalizes it, and saves the normalized data back to the directory.
    It returns the scaling factors (mean, standard deviation, min, and max after standardization)
    applied to each numerical value, excluding booleans and lists.
    
    Args:
        directory (str): The directory containing JSON files with data.
    
    Returns:
        dict: A dictionary containing the scaling factors for each feature.
    """
    # Load data
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]  # Adjusted to '.txt' based on the uploaded files
    if not files:  # Check if the directory is empty or contains no .txt files
        return "No .txt files found in the directory."
    
    data = []
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            data.append(json.load(f)[0])
    
    # Avoid proceeding if no data is found
    if not data:
        return "No data loaded from files."
    
    # Normalize data
    features = {key: [] for key in data[0].keys() if type(data[0][key]) not in [bool, list]}
    features.update({"centroid_x": [], "centroid_y": []})
    
    for entry in data:
        for key, value in entry.items():
            if key in features:
                features[key].append(value)
            elif key == "centroid":
                features["centroid_x"].append(value[0])
                features["centroid_y"].append(value[1])
    
    scale_factors = {}
    for key, values in features.items():
        values = np.array(values)
        mean = values.mean() if values.size else 0
        std = values.std() if values.size else 1
        values_std = (values - mean) / std if std != 0 else values - mean
        min_val, max_val = values_std.min(), values_std.max()
        values_norm = 2 * (values_std - min_val) / (max_val - min_val) - 1 if max_val - min_val != 0 else values_std
        features[key] = values_norm
        scale_factors[key] = {'mean': mean, 'std': std, 'min_after_std': min_val, 'max_after_std': max_val}
    
    # Update data with normalized values
    for i, entry in enumerate(data):
        for key in entry:
            if key in features:
                entry[key] = features[key][i]
            elif key == "centroid":
                entry["centroid"][0] = features["centroid_x"][i]
                entry["centroid"][1] = features["centroid_y"][i]
    
    # Save normalized data
    for entry, file in zip(data, files):
        with open(os.path.join(directory, file), 'w') as f:
            json.dump([entry], f, indent=4)
    
    return scale_factors

     
def normalise_json(input_dir, scale_factors_filename):
    # Buscar el primer archivo .txt en el directorio
    json_file = next((f for f in os.listdir(input_dir) if f.endswith('.txt')), None)
    if not json_file:
        raise FileNotFoundError("No .txt file found in the directory.")
    json_path = os.path.join(input_dir, json_file)
    with open(json_path, 'r') as file:
        data = json.load(file)
    with open(scale_factors_filename, 'r') as file:
        scale_factors = json.load(file)
    
    for entry in data:
        for key, value in entry.items():
            if key == 'centroid':
                x_key, y_key = 'centroid_x', 'centroid_y'
                entry[key][0] = normalize_value(entry[key][0], scale_factors.get(x_key, {}))
                entry[key][1] = normalize_value(entry[key][1], scale_factors.get(y_key, {}))
            else:
                if key in scale_factors and isinstance(value, (int, float)):
                    entry[key] = normalize_value(value, scale_factors[key])
    with open(json_path, 'w') as file:
        json.dump(data, file, indent=4)
def normalize_value(value, factor):
    if not factor:  
        return value
    norm_value = (value - factor['mean']) / factor['std'] if factor['std'] else value - factor['mean']
    return 2 * (norm_value - factor['min_after_std']) / (factor['max_after_std'] - factor['min_after_std']) - 1 if factor['max_after_std'] - factor['min_after_std'] else norm_value


           
def normalize_gt_labels(directory):
    """
    Loads tabular data from text files, normalizes it, and then scales it to the range [-1, 1].
    It saves the normalized and scaled data back to the directory and returns the scale factors.
    
    Args:
        directory (str): The directory containing text files with tabular data.
    
    Returns:
        dict: A dictionary containing the scale factors for each of x, y, and z including min and max after normalization.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    data = []
    
    for file in files:
        with open(os.path.join(directory, file), 'r') as f:
            try:
                line = f.readline().strip()
                x, y, z = map(float, line.split())
                data.append({'x': x, 'y': y, 'z': z})
            except ValueError:
                continue
    
    if not data:
        return "No valid tabular data files found in the directory."
    
    features = {'x': [], 'y': [], 'z': []}
    for entry in data:
        for key in features:
            features[key].append(entry[key])
    
    scale_factors = {}
    for key, values in features.items():
        values = np.array(values)
        mean = values.mean()
        std = values.std()
        values_std = (values - mean) / std if std != 0 else values - mean
        min_val, max_val = values_std.min(), values_std.max()
        values_scaled = 2 * (values_std - min_val) / (max_val - min_val) - 1 if max_val - min_val != 0 else values_std
        scale_factors[key] = {'mean': mean, 'std': std, 'min': min_val, 'max': max_val}
        for i, entry in enumerate(data):
            entry[key] = values_scaled[i]
    
    for entry, file in zip(data, files):
        with open(os.path.join(directory, file), 'w') as f:
            line = f"{entry['x']} {entry['y']} {entry['z']}\n"
            f.write(line)
    
    return scale_factors


def reconstruct_output2(scale_factors_filename, tensor_values):
    """
    Reconstructs the original scale of values given the path to a file containing scale factors,
    min, max, and a tensor of normalized and scaled values, maintaining autograd compatibility.
    
    Args:
        scale_factors_filename (str): The path to the file containing the scale factors.
        tensor_values (torch.Tensor): A tensor containing the normalized and scaled values for each dimension (x, y, z or x, y).
    
    Returns:
        torch.Tensor: A tensor containing the values in their original scale for each dimension, maintaining the input tensor's format.
    """
    # Load scale factors from the file
    with open(scale_factors_filename, 'r') as file:
        scale_factors = json.load(file)

    # Prepare a tensor to hold the reconstructed values, preserving the input format
    original_values = torch.zeros_like(tensor_values)

    keys = ['x', 'y', 'z'][:len(tensor_values)]
    for i, key in enumerate(keys):
        if key in scale_factors:
            min_val = scale_factors[key]['min']
            max_val = scale_factors[key]['max']
            # Apply inverse scaling
            value_std = ((tensor_values[i] + 1) / 2 * (max_val - min_val)) + min_val
            # Compute the original value
            if scale_factors[key]['std'] != 0:
                original_values[i] = (value_std - scale_factors[key]['mean']) / scale_factors[key]['std']
            else:
                original_values[i] = value_std - scale_factors[key]['mean']
        else:
            original_values[i] = tensor_values[i]

    return original_values



#predict_labels(PRED_MASK_DIR, SAVING_FOLDER)
#escalas_labels= normalise_predicted_data(LABELS_DIR)
#escalas_labels= normalise_json(LABELS_DIR)
#escalas_labels_gt=normalize_gt_labels(LABELS_GT_DIR)
#print(escalas_labels)