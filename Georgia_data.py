#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shutil
import os
import cv2
import numpy as np
import glob
from mtcnn import MTCNN
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
import re
import math

# Paths
FOLDER_PATHS = "C:/Users/user/Downloads/gt_db/gt_db"
OUTPUT_DIR = "C:/Users/user/Downloads/gt_db/preprocessed_images_ci"
DEBUG_DIR = "C:/Users/user/Downloads/gt_db/debug_images_ci"
# Clear the output directory
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clear the debug directory
if os.path.exists(DEBUG_DIR):
    shutil.rmtree(DEBUG_DIR)
os.makedirs(DEBUG_DIR, exist_ok=True)


# In[2]:


import os
import cv2
import numpy as np
import glob
from mtcnn import MTCNN
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
import re
import math

# Paths
FOLDER_PATHS = "C:/Users/user/Downloads/gt_db/gt_db"
OUTPUT_DIR = "C:/Users/user/Downloads/gt_db/preprocessed_images_ci"
DEBUG_DIR = "C:/Users/user/Downloads/gt_db/debug_images_ci"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Initialize detectors
mtcnn_detector = MTCNN()

def extract_label_from_filename(folder_name, filename):
    match = re.match(r'(\d+).jpg', filename)  
    return f"{folder_name}_{match.group(1)}" if match else "unknown"

def align_face(image, landmarks):
    """ Aligns face based on eye positions using affine transformation. """
    left_eye = np.array(landmarks['left_eye'])
    right_eye = np.array(landmarks['right_eye'])

    # Compute the center between the two eyes
    eye_center = (left_eye + right_eye) / 2.0
    
    # Compute the rotation angle using arctan2
    delta_x = right_eye[0] - left_eye[0]
    delta_y = right_eye[1] - left_eye[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Get rotation matrix and apply affine transformation
    rot_matrix = cv2.getRotationMatrix2D(tuple(eye_center), angle, scale=1)
    aligned_image = cv2.warpAffine(image, rot_matrix, (image.shape[1], image.shape[0]))
    
    return aligned_image

def detect_faces(image):
    """ Detects faces using MTCNN. Returns bounding box and landmarks. """
    detections = mtcnn_detector.detect_faces(image)
    return detections if detections else None

def is_duplicate(image, output_dir, threshold=0.9):
    """ Checks if an image is a duplicate using SSIM. """
    image_resized = cv2.resize(image, (100, 100))
    existing_files = os.listdir(output_dir)
    
    for filename in existing_files[:min(10, len(existing_files))]:  
        existing_image = cv2.imread(os.path.join(output_dir, filename), cv2.IMREAD_GRAYSCALE)
        if existing_image is not None:
            existing_resized = cv2.resize(existing_image, (100, 100))
            similarity = ssim(image_resized, existing_resized)
            if similarity >= threshold:
                return True
    return False

def preprocess_image(image_path, folder_name):
    """ Loads, detects face, aligns, resizes, and extracts HOG features. """
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load {image_path}")
            return None
        
        original_filename = os.path.basename(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detect_faces(image_rgb)

        if detections:
            face = detections[0]
            x, y, w, h = map(int, face['box'])
            x, y = max(0, x), max(0, y)

            if 'keypoints' in face:
                image_rgb = align_face(image_rgb, face['keypoints'])

            cropped_face = image_rgb[y:y+h, x:x+w]
            if cropped_face.size == 0:
                print(f"Warning: Invalid face crop in {image_path}")
                return None
            resized_face = cv2.resize(cropped_face, (224, 224), interpolation=cv2.INTER_AREA)
        else:
            resized_face = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_AREA)
            print(f"Warning: No face detected in {image_path}")

        gray_face = cv2.cvtColor(resized_face, cv2.COLOR_BGR2GRAY)
        equalized_face = cv2.equalizeHist(gray_face)
        normalized_face = cv2.normalize(equalized_face, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        label = extract_label_from_filename(folder_name, original_filename)
        output_file = os.path.join(OUTPUT_DIR, f"{label}.jpg")
        
        if not is_duplicate(normalized_face, OUTPUT_DIR):
            cv2.imwrite(output_file, normalized_face)
            return hog(normalized_face, orientations=9, pixels_per_cell=(8, 8), 
                       cells_per_block=(2, 2), block_norm='L2-Hys', feature_vector=True)
        else:
            print(f"Duplicate skipped: {image_path}")
            return None
    except Exception as e:
        print(f"Failed to preprocess {image_path}: {e}")
        return None

# Process dataset
features, labels = [], []
total_images, failed_images = 0, 0

for folder in os.listdir(FOLDER_PATHS):
    folder_path = os.path.join(FOLDER_PATHS, folder)
    if os.path.isdir(folder_path):
        for filename in glob.glob(os.path.join(folder_path, "*.jpg")):
            total_images += 1
            hog_feature = preprocess_image(filename, folder)
            if hog_feature is not None:
                features.append(hog_feature)
                labels.append(folder)
            else:
                failed_images += 1

print(f"\nTotal Processed: {total_images}, Failed: {failed_images}, Features Shape: {np.array(features).shape}")


# In[3]:


import numpy as np
from scipy.stats import rankdata
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

# Normalize scores
def normalize_scores(scores):
    min_score = np.min(scores)
    max_score = np.max(scores)
    return (scores - min_score) / (max_score - min_score)

# Fisher Score
def fisher_score(features, labels):
    unique_classes = np.unique(labels)
    n_features = features.shape[1]
    scores = np.zeros(n_features)

    for feature_idx in range(n_features):
        feature_values = features[:, feature_idx]
        overall_mean = np.mean(feature_values)
        
        between_class_var = 0
        within_class_var = 0
        
        for cls in unique_classes:
            cls_values = feature_values[labels == cls]
            cls_mean = np.mean(cls_values)
            cls_size = len(cls_values)
            
            between_class_var += cls_size * (cls_mean - overall_mean) ** 2
            within_class_var += np.sum((cls_values - cls_mean) ** 2)
        
        if within_class_var != 0:
            scores[feature_idx] = between_class_var / within_class_var
    
    return scores

# ReliefF
def relief_f(features, labels, n_neighbors=10):
    from sklearn.metrics.pairwise import pairwise_distances

    n_samples, n_features = features.shape
    scores = np.zeros(n_features)
    distances = pairwise_distances(features)
    np.fill_diagonal(distances, np.inf)

    for i in range(n_samples):
        same_class_mask = labels == labels[i]
        nearest_same_class = np.argsort(distances[i][same_class_mask])[:n_neighbors]
        
        different_class_mask = labels != labels[i]
        nearest_diff_class = np.argsort(distances[i][different_class_mask])[:n_neighbors]
        
        for feature_idx in range(n_features):
            scores[feature_idx] += np.sum(
                np.abs(features[i, feature_idx] - features[nearest_diff_class, feature_idx])
            )
            scores[feature_idx] -= np.sum(
                np.abs(features[i, feature_idx] - features[nearest_same_class, feature_idx])
            )
    
    scores /= (n_samples * n_neighbors)
    return scores

# VIKOR Ranking
def vikor_ranking(fisher_scores, relieff_scores, v=0.5):
    fisher_norm = normalize_scores(fisher_scores)
    relieff_norm = normalize_scores(relieff_scores)
    
    S = np.sum(np.vstack([fisher_norm, relieff_norm]), axis=0)
    R = np.max(np.vstack([fisher_norm, relieff_norm]), axis=0)
    
    S_min, S_max = np.min(S), np.max(S)
    R_min, R_max = np.min(R), np.max(R)
    
    Q = v * (S - S_min) / (S_max - S_min) + (1 - v) * (R - R_min) / (R_max - R_min)
    return rankdata(Q, method="min")

# TOPSIS Ranking
def topsis_ranking(fisher_scores, relieff_scores):
    fisher_norm = normalize_scores(fisher_scores)
    relieff_norm = normalize_scores(relieff_scores)
    
    ideal = np.max(np.vstack([fisher_norm, relieff_norm]), axis=0)
    anti_ideal = np.min(np.vstack([fisher_norm, relieff_norm]), axis=0)
    
    dist_to_ideal = np.sqrt(np.sum((np.vstack([fisher_norm, relieff_norm]) - ideal) ** 2, axis=0))
    dist_to_anti_ideal = np.sqrt(np.sum((np.vstack([fisher_norm, relieff_norm]) - anti_ideal) ** 2, axis=0))
    
    topsis_score = dist_to_anti_ideal / (dist_to_ideal + dist_to_anti_ideal)
    return rankdata(-topsis_score, method="min")

# CODAS Ranking
def codas_ranking(fisher_scores, relieff_scores):
    fisher_norm = normalize_scores(fisher_scores)
    relieff_norm = normalize_scores(relieff_scores)
    
    euclidean_dist = np.sqrt(np.sum((np.vstack([fisher_norm, relieff_norm]) - 0) ** 2, axis=0))
    taxicab_dist = np.sum(np.abs(np.vstack([fisher_norm, relieff_norm]) - 0), axis=0)
    
    codas_score = euclidean_dist + 0.5 * taxicab_dist
    return rankdata(codas_score, method="min")

# Aggregate Rankings
def aggregate_rankings(fisher_scores, relieff_scores):
    vikor = vikor_ranking(fisher_scores, relieff_scores)
    topsis = topsis_ranking(fisher_scores, relieff_scores)
    codas = codas_ranking(fisher_scores, relieff_scores)
    
    final_ranking = (vikor + topsis + codas) / 3
    return final_ranking

# Create Feature Subsets
def create_feature_subsets(features, final_ranking, percentages=[0.25, 0.5, 0.75, 1.0]):
    n_features = features.shape[1]
    sorted_indices = np.argsort(final_ranking)
    subsets = {}
    
    for perc in percentages:
        top_k = int(perc * n_features)
        selected_features = features[:, sorted_indices[:top_k]]
        subsets[perc] = selected_features
    
    return subsets

# Evaluate Feature Subsets
def evaluate_subsets(feature_subsets, labels, model=SVC(kernel='linear'), cv=5):
    results = {}
    for perc, subset in feature_subsets.items():
        scores = cross_val_score(model, subset, labels, cv=cv)
        results[perc] = np.mean(scores)
    return results


# Convert feature list to NumPy array
features_array = np.array(features)  
labels_array = np.array(labels)

# Ensure labels are numerical if needed
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
labels_array = label_encoder.fit_transform(labels_array)  # Convert string labels to numbers

# Compute Fisher and ReliefF Scores
fisher_scores = fisher_score(features_array, labels_array)
relieff_scores = relief_f(features_array, labels_array)

print("\nFisher Scores:\n", fisher_scores)
print("\nReliefF Scores:\n", relieff_scores)


print("\nFisher Scores:\n", fisher_scores)
print("\nReliefF Scores:\n", relieff_scores)

# Combine Rankings
final_ranking = aggregate_rankings(fisher_scores, relieff_scores)
print("\nHybrid Ranking (Final Aggregated Ranking):\n", final_ranking)

# Concatenate Transformed Features
def concatenate_transformed_features(features, final_ranking, top_k):
    sorted_indices = np.argsort(final_ranking)
    selected_features = features[:, sorted_indices[:top_k]]
    return selected_features

top_k_features = int(0.50 * features_array.shape[1])  # Using top 50% features 
concatenated_features = concatenate_transformed_features(features_array, final_ranking, top_k_features)

# Create Subsets
feature_subsets = {
    perc: concatenate_transformed_features(features_array, final_ranking, int(perc * features_array.shape[1]))
    for perc in [0.25, 0.5, 0.75, 1.0]
}

# Evaluate Subsets
subset_results = evaluate_subsets(feature_subsets, labels_array)
for perc, score in subset_results.items():
    print(f"Top {int(perc * 100)}% features: Accuracy = {score:.4f}")



# In[4]:


import numpy as np
from sklearn.model_selection import train_test_split

def split_dataset_by_samples(features, labels, n_samples_per_class, random_state=42):
    np.random.seed(random_state)
    unique_classes = np.unique(labels)
    train_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        train_indices.extend(cls_indices[:n_samples_per_class])
        test_indices.extend(cls_indices[n_samples_per_class:])

    return features[train_indices], features[test_indices], labels[train_indices], labels[test_indices]

# Example usage:
n_samples_per_class = 3  # Change this to 5, 7, or 9 as needed
X_train, X_test, y_train, y_test = split_dataset_by_samples(features_array, labels_array, n_samples_per_class)


# In[5]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_classifier(feature_subsets, labels, n_samples_per_class, classifier, classifier_name):
    results = {}
    
    for perc, subset in feature_subsets.items():
        X_train, X_test, y_train, y_test = split_dataset_by_samples(subset, labels, n_samples_per_class)
        
        clf = classifier
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Ensure all labels exist in the classification report
        unique_labels = np.unique(labels)
        report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=1, output_dict=True)
        
        results[perc] = {
            "accuracy": accuracy,
            "classification_report": report
        }
        print(f"{classifier_name} - Top {int(perc * 100)}% features: Accuracy = {accuracy:.4f}")
    
    return results

# Ensure feature subsets are scaled before classification
scaler = StandardScaler()
feature_subsets = {perc: scaler.fit_transform(subset) for perc, subset in feature_subsets.items()}

# Evaluate KNN
print("\nEvaluating KNN...")
knn_results = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=3, classifier=KNeighborsClassifier(n_neighbors=5), classifier_name="KNN"
)

# Evaluate SVM
print("\nEvaluating SVM...")
svm_results = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=3, classifier=SVC(kernel='linear', C=0.1, random_state=42), classifier_name="SVM"
)


# In[6]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

def train_and_evaluate_classifier(feature_subsets, labels, n_samples_per_class, classifier, classifier_name):
    results = {}
    
    for perc, subset in feature_subsets.items():
        X_train, X_test, y_train, y_test = split_dataset_by_samples(subset, labels, n_samples_per_class)
        
        clf = classifier
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Ensure all labels exist in the classification report
        unique_labels = np.unique(labels)
        report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=1, output_dict=True)
        
        results[perc] = {
            "accuracy": accuracy,
            "classification_report": report
        }
        print(f"{classifier_name} - Top {int(perc * 100)}% features: Accuracy = {accuracy:.4f}")
    
    return results

# Ensure feature subsets are scaled before classification
scaler = StandardScaler()
feature_subsets = {perc: scaler.fit_transform(subset) for perc, subset in feature_subsets.items()}

# Evaluate KNN
print("\nEvaluating KNN...")
knn_results = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=3, classifier=KNeighborsClassifier(n_neighbors=5), classifier_name="KNN"
)

# Evaluate SVM
print("\nEvaluating SVM...")
svm_results = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=3, classifier=SVC(kernel='linear', C=0.1, random_state=42), classifier_name="SVM"
)


# In[7]:


# Change n_samples_per_class to 5, 7, or 9 for different evaluations
knn_results_5 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=5, classifier=KNeighborsClassifier(n_neighbors=5), classifier_name="KNN"
)

svm_results_5 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=5, classifier=SVC(kernel='linear', C=0.1, random_state=42), classifier_name="SVM"
)

# Repeat for 7 and 9 samples per class
knn_results_7 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=7, classifier=KNeighborsClassifier(n_neighbors=5), classifier_name="KNN"
)

svm_results_7 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=7, classifier=SVC(kernel='linear', C=0.1, random_state=42), classifier_name="SVM"
)

knn_results_9 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=9, classifier=KNeighborsClassifier(n_neighbors=5), classifier_name="KNN"
)

svm_results_9 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=9, classifier=SVC(kernel='linear', C=0.1, random_state=42), classifier_name="SVM"
)


# In[8]:


knn_results_3 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=3, classifier=KNeighborsClassifier(n_neighbors=5), classifier_name="KNN"
)


svm_results_3 = train_and_evaluate_classifier(
    feature_subsets, labels_array, n_samples_per_class=3, classifier=SVC(kernel='linear', C=0.1, random_state=42), classifier_name="SVM"
)


# In[9]:


# Example structure for storing and printing the results
results = {
    'KNN': {
        3: knn_results_3,
        5: knn_results_5,
        7: knn_results_7,
        9: knn_results_9,
    },
    'SVM': {
        3: svm_results_3,
        5: svm_results_5,
        7: svm_results_7,
        9: svm_results_9,
    }
}

for classifier, samples_results in results.items():
    print(f"\n--- {classifier} Results ---")
    for n_samples, metrics in samples_results.items():
        print(f"\n--- Training with {n_samples} Samples per Class ---")
        for perc, score in metrics.items():
            print(f"Top {int(perc * 100)}% features: Accuracy = {score['accuracy']:.4f}")
            print("Classification Report:")
            for class_label, scores in score['classification_report'].items():
                if isinstance(scores, dict):
                    print(f"Class {class_label}: Precision = {scores['precision']:.4f}, Recall = {scores['recall']:.4f}, F1 = {scores['f1-score']:.4f}")


# In[ ]:





# In[10]:


from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier  # Import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np

def split_dataset(features, labels, test_size=0.3, random_state=42):
    return train_test_split(features, labels, test_size=test_size, stratify=labels, random_state=random_state)

def train_and_evaluate_classifier(feature_subsets, labels, classifier, classifier_name):
    results = {}
    
    for perc, subset in feature_subsets.items():
        X_train, X_test, y_train, y_test = split_dataset(subset, labels)
        
        clf = classifier
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Ensure all labels exist in the classification report
        unique_labels = np.unique(labels)
        report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=1, output_dict=True)
        
        results[perc] = {
            "accuracy": accuracy,
            "classification_report": report
        }
        print(f"{classifier_name} - Top {int(perc * 100)}% features: Accuracy = {accuracy:.4f}")
    
    return results

# Ensure feature subsets are scaled before classification
scaler = StandardScaler()
feature_subsets = {perc: scaler.fit_transform(subset) for perc, subset in feature_subsets.items()}

# Evaluate KNN
print("\nEvaluating KNN...")
knn_results = train_and_evaluate_classifier(
    feature_subsets, labels_array, KNeighborsClassifier(n_neighbors=5), "KNN"
)

# Evaluate SVM
print("\nEvaluating SVM...")
svm_results = train_and_evaluate_classifier(
    feature_subsets, labels_array, SVC(kernel='linear', C=0.1, random_state=42), "SVM"
)

def display_classification_reports(results, classifier_name):
    print(f"\n--- {classifier_name} Results ---")
    for perc, metrics in results.items():
        print(f"\n--- Top {int(perc * 100)}% Features ---")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print("Classification Report:")
        for class_label, scores in metrics['classification_report'].items():
            if isinstance(scores, dict):
                print(f"Class {class_label}: Precision = {scores['precision']:.4f}, Recall = {scores['recall']:.4f}, F1 = {scores['f1-score']:.4f}")

# Display results
display_classification_reports(knn_results, "KNN")
display_classification_reports(svm_results, "SVM")


# In[11]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}

# Perform Grid Search
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
grid_search.fit(feature_subsets[1.0], labels_array)  # Using full feature set

print(f"Best KNN parameters: {grid_search.best_params_}")


# In[12]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define parameter grid for SVM
param_grid_svm = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Different kernel types
    'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf' and 'poly'
}

# Perform Grid Search for SVM
grid_search_svm = GridSearchCV(SVC(), param_grid_svm, cv=5)
grid_search_svm.fit(feature_subsets[1.0], labels_array)  # Using full feature set

# Print best parameters
print(f"Best SVM parameters: {grid_search_svm.best_params_}")


# In[13]:


import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Define functions (normalize_scores, fisher_score, relief_f, vikor_ranking, topsis_ranking, codas_ranking, aggregate_rankings, create_feature_subsets, evaluate_subsets)

def split_dataset_by_samples(features, labels, n_samples_per_class, random_state=42):
    np.random.seed(random_state)
    unique_classes = np.unique(labels)
    train_indices = []
    test_indices = []

    for cls in unique_classes:
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        train_indices.extend(cls_indices[:n_samples_per_class])
        test_indices.extend(cls_indices[n_samples_per_class:])
    
    return features[train_indices], features[test_indices], labels[train_indices], labels[test_indices]

def train_and_evaluate_classifier(feature_subsets, labels, n_samples_per_class, classifier, classifier_name):
    results = {}
    
    for perc, subset in feature_subsets.items():
        X_train, X_test, y_train, y_test = split_dataset_by_samples(subset, labels, n_samples_per_class)
        
        clf = classifier
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        unique_labels = np.unique(labels)
        report = classification_report(y_test, y_pred, labels=unique_labels, zero_division=1, output_dict=True)
        
        results[perc] = {
            "accuracy": accuracy,
            "classification_report": report
        }
        print(f"{classifier_name} - Top {int(perc * 100)}% features: Accuracy = {accuracy:.4f}")
    
    return results

# Define features_array, labels_array, and other preprocessing steps

# Define sample sizes and initialize results storage
sample_sizes = [3, 5, 7, 9]
results = {
    'KNN': {},
    'SVM': {}
}

# Train and evaluate classifiers for each sample size
for n_samples in sample_sizes:
    print(f"\nEvaluating with {n_samples} training samples per class...")
    
    # Evaluate KNN
    print(f"\nTraining and Evaluating KNN with {n_samples} samples per class...")
    knn_results = train_and_evaluate_classifier(
        feature_subsets, labels_array, n_samples_per_class=n_samples, classifier=KNeighborsClassifier(n_neighbors=5), classifier_name="KNN"
    )
    results['KNN'][n_samples] = knn_results
    
    # Evaluate SVM
    print(f"\nTraining and Evaluating SVM with {n_samples} samples per class...")
    svm_results = train_and_evaluate_classifier(
        feature_subsets, labels_array, n_samples_per_class=n_samples, classifier=SVC(kernel='linear', C=0.1, random_state=42), classifier_name="SVM"
    )
    results['SVM'][n_samples] = svm_results

# Display the results
def display_results(results):
    for classifier, samples_results in results.items():
        print(f"\n--- {classifier} Results ---")
        for n_samples, metrics in samples_results.items():
            print(f"\n--- Training with {n_samples} Samples per Class ---")
            for perc, score in metrics.items():
                print(f"Top {int(perc * 100)}% features: Accuracy = {score['accuracy']:.4f}")
                print("Classification Report:")
                for class_label, scores in score['classification_report'].items():
                    if isinstance(scores, dict):
                        print(f"Class {class_label}: Precision = {scores['precision']:.4f}, Recall = {scores['recall']:.4f}, F1 = {scores['f1-score']:.4f}")

display_results(results)

