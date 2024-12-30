import torch
import cv2
import numpy as np
from model.model import QuickDraw  # Make sure to import the model correctly

def create_tensor_mask(mask_img, input_size=28):
    """
    Converts a mask (grayscale image) into a tensor input for the QuickDraw model.
    
    Parameters:
    - mask_img: The mask image, should be a grayscale image (1 channel).
    - input_size: The height and width size of the input image (default is 28).
    
    Returns:
    - tensor_mask: A tensor suitable for model input, with shape (1, 1, input_size, input_size).
    """
    
    # Check if the input image is grayscale (1 channel)
    if len(mask_img.shape) != 2:
        raise ValueError("The mask image should be a grayscale image (1 channel).")
    
    # Resize the mask image to match the input_size
    mask_resized = cv2.resize(mask_img, (input_size, input_size))

    # Convert the resized mask image to a tensor
    tensor_mask = torch.tensor(mask_resized, dtype=torch.float32)

    # Normalize the tensor mask to the range [0, 1]
    tensor_mask = tensor_mask / 255.0

    # Add batch and channel dimensions (shape: (1, 1, input_size, input_size))
    tensor_mask = tensor_mask.unsqueeze(0).unsqueeze(0)

    return tensor_mask

def classifier(mask, model_path='D:/HUST/Project1/Camera Draw/train/trained_models/50epochs.pth'):
    """
    Classifies the given mask using a pre-trained QuickDraw model.

    Parameters:
    - mask: Grayscale mask image (e.g., thresholded or resized to match the model input size).
    - model_path: Path to the pre-trained model file.

    Returns:
    - predicted_class: The predicted class index from the QuickDraw model.
    """
    # Ensure mask is resized to 28x28
    mask_resized = cv2.resize(mask, (28, 28))
    
    # Convert the mask to a tensor
    mask_tensor = create_tensor_mask(mask_resized)
    
    # Load the pre-trained QuickDraw model
    model = QuickDraw()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Perform inference
    with torch.no_grad():
        output = model(mask_tensor)
    
    # Get the predicted class
    _, predicted_class = torch.max(output.data, 1)
    
    return predicted_class.item()

def main():
    tensor = torch.load('tensor_mask.pt')
    predicted_class = classifier(tensor)
    print(f"Predicted class: {predicted_class}")
    
if __name__ == "__main__":
    main()