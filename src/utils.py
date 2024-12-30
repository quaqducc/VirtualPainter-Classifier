import cv2
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix

def get_evaluation(y_true, y_prob, list_metrics):
    """
    Calculate evaluation metrics for classification tasks.

    Parameters:
    - y_true: True labels
    - y_prob: Predicted probabilities (output from a model)
    - list_metrics: List of metrics to calculate (accuracy, loss, confusion_matrix)

    Returns:
    - output: Dictionary with the requested metrics
    """
    # Get predicted class labels from predicted probabilities
    y_pred = np.argmax(y_prob, axis=-1)
    
    output = {}

    # Calculate accuracy if requested
    if 'accuracy' in list_metrics:
        output['accuracy'] = accuracy_score(y_true, y_pred)

    # Calculate log loss if requested
    if 'loss' in list_metrics:
        try:
            output['loss'] = log_loss(y_true, y_prob)
        except ValueError:
            output['loss'] = -1  # Return -1 if log_loss calculation fails

    # Calculate confusion matrix if requested
    if 'confusion_matrix' in list_metrics:
        output['confusion_matrix'] = str(confusion_matrix(y_true, y_pred))
    
    return output


def get_images(path, classes):
    """
    Load images from a folder for each class.

    Parameters:
    - path: Path to the directory containing images
    - classes: List of class names for which to load images

    Returns:
    - images: List of loaded images
    """
    images = []
    for item in classes:
        image = cv2.imread(f"{path}/{item}.png", cv2.IMREAD_UNCHANGED)
        if image is not None:
            images.append(image)
    return images


def get_overlay(bg_image, fg_image, sizes=(40, 40)):
    """
    Overlay a transparent image (fg_image) onto a background image (bg_image).

    Parameters:
    - bg_image: Background image on which to overlay
    - fg_image: Foreground image with transparency (alpha channel)
    - sizes: Resize dimensions for the foreground image

    Returns:
    - image: Combined image after overlaying
    """
    # Resize foreground image
    fg_image = cv2.resize(fg_image, sizes)
    
    # Split the foreground image into color (BGR) and alpha channels
    fg_mask = fg_image[:, :, 3:]  # Extract alpha channel
    fg_image = fg_image[:, :, :3]  # Extract color channels
    
    # Create the inverse of the alpha channel as background mask
    bg_mask = 255 - fg_mask
    bg_image = bg_image / 255  # Normalize background image
    fg_image = fg_image / 255  # Normalize foreground image
    fg_mask = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) / 255  # Normalize foreground mask
    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR) / 255  # Normalize background mask
    
    # Combine the images by blending foreground and background using the masks
    image = cv2.addWeighted(bg_image * bg_mask, 255, fg_image * fg_mask, 255, 0).astype(np.uint8)
    return image


# Example usage
if __name__ == '__main__':
    images = get_images("../images", ["apple", "star"])
    print(images[0].shape)
    print(np.max(images[0]))
