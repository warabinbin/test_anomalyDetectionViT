import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
from PIL import Image
from tqdm import tqdm
import timm
import torch
import torch.nn as nn
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

# Import your existing anomaly detection functions
# This assumes we're importing from your anomaly_detection_vit.py file
from anomaly_detection_vit import ImprovedViTFeatureExtractor, load_images_from_folder, train_transform, test_transform

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def evaluate_anomaly_detection(model_func, train_folder, normal_test_folder, abnormal_test_folder,
                              threshold=None, threshold_method='percentile', use_pca=True):
    """
    Evaluate anomaly detection model using standard metrics
    
    Args:
        model_func: Function that performs anomaly detection
        train_folder: Folder with normal training images
        normal_test_folder: Folder with normal test images
        abnormal_test_folder: Folder with abnormal test images
        threshold: Anomaly detection threshold (None for automatic)
        threshold_method: Method for threshold determination
        use_pca: Whether to use PCA for dimension reduction
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load the model
    model = ImprovedViTFeatureExtractor().to(device)
    model.eval()
    
    # Load normal training images (with augmentation)
    print("Loading normal training images...")
    train_imgs, _ = load_images_from_folder(train_folder, transform=train_transform, 
                                           augment=True, num_augmentations=4)
    
    if len(train_imgs) == 0:
        print(f"Error: No normal training images found in {train_folder}")
        return
    
    print(f"Number of training images after augmentation: {len(train_imgs)}")
    
    # Extract features from normal training images
    print("Extracting features from normal training images...")
    train_features = []
    batch_size = 16
    
    with torch.no_grad():
        for i in range(0, len(train_imgs), batch_size):
            batch = train_imgs[i:i+batch_size].to(device)
            feat = model(batch)
            train_features.append(feat.cpu().numpy())
    
    train_features = np.vstack(train_features)
    
    # PCA (optional)
    if use_pca:
        print("Reducing feature dimensions with PCA...")
        pca = PCA(n_components=0.95)  # Keep 95% of variance
        train_features = pca.fit_transform(train_features)
        print(f"Feature dimensions after PCA: {train_features.shape[1]}")
    
    # Use kNN for small datasets, Mahalanobis distance for larger ones
    use_knn = len(train_features) < 200
    
    if use_knn:
        print(f"Using kNN due to small dataset size ({len(train_features)} samples)")
        nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
        nn_model.fit(train_features)
    else:
        try:
            print("Computing robust covariance estimation...")
            # Adjust support fraction based on dataset size
            support_fraction = min(0.6, (len(train_features) - 1) / len(train_features))
            robust_cov = MinCovDet(support_fraction=support_fraction).fit(train_features)
        except ValueError as e:
            print(f"MinCovDet error: {e}")
            print("Falling back to kNN method")
            use_knn = True
            nn_model = NearestNeighbors(n_neighbors=1, metric='euclidean')
            nn_model.fit(train_features)
    
    # Calculate distances for normal training samples (for threshold determination)
    if use_knn:
        distances, _ = nn_model.kneighbors(train_features)
        normal_distances = [dist[0] for dist in distances]
    else:
        normal_distances = robust_cov.mahalanobis(train_features)
    
    # Normalize scores
    normal_scores = normalize_scores(normal_distances)
    
    # Determine threshold if not provided
    if threshold is None:
        threshold = compute_adaptive_threshold(normal_scores, method=threshold_method)
    
    # Process normal test images
    print("Processing normal test images...")
    normal_test_imgs, normal_filenames = load_images_from_folder(normal_test_folder, transform=test_transform)
    
    if len(normal_test_imgs) == 0:
        print(f"Error: No normal test images found in {normal_test_folder}")
        return
    
    normal_test_features = []
    with torch.no_grad():
        for i in range(0, len(normal_test_imgs), batch_size):
            batch = normal_test_imgs[i:i+batch_size].to(device)
            feat = model(batch)
            normal_test_features.append(feat.cpu().numpy())
    
    normal_test_features = np.vstack(normal_test_features)
    
    # PCA projection (if used)
    if use_pca:
        normal_test_features = pca.transform(normal_test_features)
    
    # Calculate anomaly scores for normal test images
    if use_knn:
        distances, _ = nn_model.kneighbors(normal_test_features)
        normal_test_scores = [dist[0] for dist in distances]
    else:
        normal_test_scores = robust_cov.mahalanobis(normal_test_features)
    
    # Normalize scores
    normal_test_scores = normalize_scores(normal_test_scores)
    
    # Process abnormal test images
    print("Processing abnormal test images...")
    abnormal_test_imgs, abnormal_filenames = load_images_from_folder(abnormal_test_folder, transform=test_transform)
    
    if len(abnormal_test_imgs) == 0:
        print(f"Error: No abnormal test images found in {abnormal_test_folder}")
        return
    
    abnormal_test_features = []
    with torch.no_grad():
        for i in range(0, len(abnormal_test_imgs), batch_size):
            batch = abnormal_test_imgs[i:i+batch_size].to(device)
            feat = model(batch)
            abnormal_test_features.append(feat.cpu().numpy())
    
    abnormal_test_features = np.vstack(abnormal_test_features)
    
    # PCA projection (if used)
    if use_pca:
        abnormal_test_features = pca.transform(abnormal_test_features)
    
    # Calculate anomaly scores for abnormal test images
    if use_knn:
        distances, _ = nn_model.kneighbors(abnormal_test_features)
        abnormal_test_scores = [dist[0] for dist in distances]
    else:
        abnormal_test_scores = robust_cov.mahalanobis(abnormal_test_features)
    
    # Normalize scores
    abnormal_test_scores = normalize_scores(abnormal_test_scores)
    
    # Combine all test results
    all_scores = np.concatenate([normal_test_scores, abnormal_test_scores])
    all_labels = np.concatenate([np.zeros(len(normal_test_scores)), np.ones(len(abnormal_test_scores))])
    all_filenames = normal_filenames + abnormal_filenames
    
    # Make predictions based on threshold
    predictions = (all_scores > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, predictions)
    precision = precision_score(all_labels, predictions)
    recall = recall_score(all_labels, predictions)
    f1 = f1_score(all_labels, predictions)
    conf_matrix = confusion_matrix(all_labels, predictions)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)
    
    # Create visual results
    # Plot anomaly scores
    plt.figure(figsize=(12, 6))
    
    # Sort by score
    sorted_indices = np.argsort(all_scores)
    sorted_scores = all_scores[sorted_indices]
    sorted_labels = all_labels[sorted_indices]
    sorted_filenames = [all_filenames[i] for i in sorted_indices]
    
    # Bar colors: green for normal, red for abnormal
    colors = ['red' if label == 1 else 'green' for label in sorted_labels]
    
    plt.bar(range(len(sorted_scores)), sorted_scores, color=colors)
    plt.axhline(y=threshold, color='orange', linestyle='--', label=f'Threshold ({threshold:.4f})')
    
    plt.xticks(range(len(sorted_scores)), sorted_filenames, rotation=45, ha='right')
    plt.ylabel("Anomaly Score")
    plt.title("Anomaly Detection Scores")
    plt.legend()
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('anomaly_detection_evaluation.png')
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    classes = ["Normal", "Abnormal"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations to confusion matrix
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the confusion matrix
    plt.savefig('confusion_matrix.png')
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    # Save the ROC curve
    plt.savefig('roc_curve.png')
    
    # Display all plots
    plt.show()
    
    # Print evaluation results
    print("\nEvaluation Results:")
    print(f"Threshold: {threshold:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"AUC: {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(conf_matrix)
    
    # Calculate TP, FP, TN, FN
    tn, fp, fn, tp = conf_matrix.ravel()
    
    # Calculate additional metrics
    specificity = tn / (tn + fp)
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    
    print("\nAdditional Metrics:")
    print(f"Specificity: {specificity:.4f}")
    print(f"False Positive Rate: {false_positive_rate:.4f}")
    print(f"False Negative Rate: {false_negative_rate:.4f}")
    
    # Return all metrics in a dictionary
    metrics = {
        'threshold': threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc,
        'specificity': specificity,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'confusion_matrix': conf_matrix,
        'scores': all_scores,
        'labels': all_labels
    }
    
    return metrics

# Helper functions (copied from your code to make this independent)
def normalize_scores(scores):
    """Normalize scores to handle numerical stability issues"""
    epsilon = 1e-10  # Small value for numerical stability
    
    # Min-Max normalization
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    # If score range is very small
    if max_score - min_score < epsilon:
        print("Warning: Score variation is very small. Applying alternative normalization.")
        # Standardize with mean=0, std=1
        mean = np.mean(scores)
        std = np.std(scores)
        
        # Use default std if very small
        if std < epsilon:
            std = 1.0
            print("Warning: Standard deviation is very small. Using default value.")
        
        normalized_scores = (scores - mean) / std
    else:
        # Normal Min-Max normalization
        normalized_scores = (scores - min_score) / (max_score - min_score)
    
    return normalized_scores

def compute_adaptive_threshold(normal_scores, method='percentile', **kwargs):
    """
    Compute adaptive threshold for anomaly detection
    
    Args:
        normal_scores: Array of scores from normal data
        method: Method for threshold determination ('percentile', 'zscore', 'iqr', 'gmm')
        **kwargs: Parameters for each method
    
    Returns:
        threshold value
    """
    if method == 'percentile':
        # Percentile-based threshold
        percentile = kwargs.get('percentile', 95)
        return np.percentile(normal_scores, percentile)
    
    elif method == 'zscore':
        # Z-score based outlier detection
        n_sigma = kwargs.get('n_sigma', 2.0)
        mean = np.mean(normal_scores)
        std = np.std(normal_scores)
        return mean + n_sigma * std
    
    elif method == 'iqr':
        # IQR-based outlier detection
        q1 = np.percentile(normal_scores, 25)
        q3 = np.percentile(normal_scores, 75)
        iqr = q3 - q1
        k = kwargs.get('k', 1.5)
        return q3 + k * iqr
    
    elif method == 'gmm':
        # Gaussian Mixture Model for threshold determination
        from sklearn.mixture import GaussianMixture
        n_components = kwargs.get('n_components', 2)
        
        # Reshape scores
        scores_reshaped = normal_scores.reshape(-1, 1)
        
        # Fit GMM
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(scores_reshaped)
        
        # Component means and variances
        means = gmm.means_.flatten()
        variances = gmm.covariances_.flatten()
        
        # Component with larger mean typically represents anomalies
        normal_idx = np.argmin(means)
        abnormal_idx = np.argmax(means)
        
        # Threshold: abnormal mean - n×std
        n_sigma = kwargs.get('n_sigma', 2.0)
        threshold = means[abnormal_idx] - n_sigma * np.sqrt(variances[abnormal_idx])
        
        return threshold
    
    else:
        # Default: 95th percentile
        return np.percentile(normal_scores, 95)

# Function to find optimal threshold
def find_optimal_threshold(scores, labels, metric='f1'):
    """
    Find optimal threshold by maximizing a given metric
    
    Args:
        scores: Array of anomaly scores
        labels: Ground truth labels (0 for normal, 1 for abnormal)
        metric: Metric to optimize ('f1', 'accuracy', 'precision', 'recall')
    
    Returns:
        optimal threshold value
    """
    # Sort scores
    sorted_indices = np.argsort(scores)
    sorted_scores = scores[sorted_indices]
    sorted_labels = labels[sorted_indices]
    
    best_metric_value = 0
    best_threshold = 0
    
    # Try each score as threshold
    for i in range(len(sorted_scores)):
        threshold = sorted_scores[i]
        predictions = (scores > threshold).astype(int)
        
        if metric == 'f1':
            metric_value = f1_score(labels, predictions)
        elif metric == 'accuracy':
            metric_value = accuracy_score(labels, predictions)
        elif metric == 'precision':
            metric_value = precision_score(labels, predictions)
        elif metric == 'recall':
            metric_value = recall_score(labels, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        # Update best threshold if metric improves
        if metric_value > best_metric_value:
            best_metric_value = metric_value
            best_threshold = threshold
    
    return best_threshold, best_metric_value

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate anomaly detection performance')
    parser.add_argument('--train', type=str, default='./data/train/normal', help='Path to normal training images')
    parser.add_argument('--test_normal', type=str, default='./data/test/normal', help='Path to normal test images')
    parser.add_argument('--test_abnormal', type=str, default='./data/test/abnormal', help='Path to abnormal test images')
    parser.add_argument('--threshold_method', type=str, default='percentile',
                        choices=['percentile', 'zscore', 'iqr', 'gmm', 'optimal'],
                        help='Method for threshold determination')
    parser.add_argument('--use_pca', type=bool, default=True, help='Whether to use PCA')
    parser.add_argument('--optimize_metric', type=str, default='f1',
                        choices=['f1', 'accuracy', 'precision', 'recall'],
                        help='Metric to optimize threshold for (when using optimal threshold)')
    
    args = parser.parse_args()
    
    # Run evaluation
    metrics = evaluate_anomaly_detection(
        model_func=None,  # Not used in this implementation
        train_folder=args.train,
        normal_test_folder=args.test_normal,
        abnormal_test_folder=args.test_abnormal,
        threshold=None,  # Automatic threshold
        threshold_method=args.threshold_method,
        use_pca=args.use_pca
    )
    
    # If optimal threshold is requested, use the returned scores to find it
    if args.threshold_method == 'optimal' and metrics is not None:
        optimal_threshold, optimal_metric = find_optimal_threshold(
            metrics['scores'], metrics['labels'], metric=args.optimize_metric
        )
        
        print(f"\nOptimal Threshold (maximizing {args.optimize_metric}): {optimal_threshold:.4f}")
        print(f"Optimal {args.optimize_metric} value: {optimal_metric:.4f}")
        
        # Re-run evaluation with optimal threshold
        metrics = evaluate_anomaly_detection(
            model_func=None,
            train_folder=args.train,
            normal_test_folder=args.test_normal,
            abnormal_test_folder=args.test_abnormal,
            threshold=optimal_threshold,
            threshold_method='percentile',  # Not used when threshold is provided
            use_pca=args.use_pca
        )