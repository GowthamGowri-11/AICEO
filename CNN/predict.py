"""
Single-Image Prediction Script for Dog-Cat CNN Classification.

Usage:
    python predict.py <path_to_image>
    python predict.py test_images/test_cat.jpg
    python predict.py test_images/test_dog.jpg

If no image path is provided, the script tests all sample images in CNN/test_images/.
"""

import os
import sys
import argparse
import numpy as np
from PIL import Image
import tensorflow as tf

# Suppress unnecessary TensorFlow logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "results", "cnn_model.keras")
DEFAULT_TEST_DIR = os.path.join(BASE_DIR, "test_images")
IMAGE_SIZE = (128, 128)
CLASS_NAMES = ["Cat", "Dog"]


def load_cnn_model(model_path=DEFAULT_MODEL_PATH):
    """Load the trained Keras CNN model."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at: '{model_path}'. "
            "Please ensure the model has been trained and saved under CNN/results/cnn_model.keras."
        )
    print(f"Loading trained CNN model from: {model_path}...")
    model = tf.keras.models.load_model(model_path)
    return model


def predict_image(image_path, model):
    """
    Predict whether a single image contains a Cat or a Dog.
    
    Args:
        image_path (str): Path to image file.
        model (tf.keras.Model): Loaded Keras model.
        
    Returns:
        dict: Prediction results including class label, confidence, and raw probability.
    """
    if not os.path.exists(image_path):
        print(f"\n[Error] File not found: '{image_path}'")
        return None

    try:
        # Load image and inspect properties
        with Image.open(image_path) as raw_img:
            orig_width, orig_height = raw_img.size
            img_format = raw_img.format
            img_mode = raw_img.mode
            img_rgb = raw_img.convert("RGB")
            img_resized = img_rgb.resize(IMAGE_SIZE)

        # Convert to numpy array: shape (1, 128, 128, 3), pixel range [0, 255]
        # (The model's internal Rescaling layer normalizes to [0, 1])
        img_array = np.array(img_resized)
        input_tensor = np.expand_dims(img_array, axis=0)

        # Forward pass inference
        raw_pred = float(model.predict(input_tensor, verbose=0)[0][0])
        pred_idx = 1 if raw_pred >= 0.5 else 0
        predicted_class = CLASS_NAMES[pred_idx]
        confidence = raw_pred if pred_idx == 1 else (1.0 - raw_pred)

        p_cat = (1.0 - raw_pred) * 100
        p_dog = raw_pred * 100

        # Visual confidence bar
        cat_bar_len = int(round(p_cat / 4))
        dog_bar_len = int(round(p_dog / 4))
        cat_bar = "#" * cat_bar_len + "-" * (25 - cat_bar_len)
        dog_bar = "#" * dog_bar_len + "-" * (25 - dog_bar_len)

        print("\n" + "=" * 60)
        print("          CNN INFERENCE PREDICTION REPORT")
        print("=" * 60)
        print(f"Image File:          {os.path.basename(image_path)}")
        print(f"Full Path:           {os.path.abspath(image_path)}")
        print(f"Original Resolution: {orig_width} x {orig_height} ({img_format}, mode={img_mode})")
        print(f"Model Input Size:    {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} x 3")
        print("-" * 60)
        print(f"PREDICTED CLASS:     >>> {predicted_class.upper()} <<<")
        print(f"CONFIDENCE:          {confidence * 100:.2f}%")
        print("-" * 60)
        print(f"Class Probabilities:")
        print(f"  P(Cat): {p_cat:6.2f}%  [{cat_bar}]")
        print(f"  P(Dog): {p_dog:6.2f}%  [{dog_bar}]")
        print("=" * 60)

        return {
            "image_path": image_path,
            "predicted_class": predicted_class,
            "confidence": confidence,
            "p_cat": p_cat / 100.0,
            "p_dog": raw_pred,
            "dimensions": (orig_width, orig_height)
        }

    except Exception as e:
        print(f"\n[Error] Failed to process image '{image_path}': {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Predict whether an image is a Cat or a Dog using the trained 3-Layer CNN model."
    )
    parser.add_argument(
        "image_path",
        nargs="?",
        default=None,
        help="Path to the image to classify (e.g., test_images/test_cat.jpg)"
    )
    parser.add_argument(
        "--model",
        dest="model_path",
        default=DEFAULT_MODEL_PATH,
        help=f"Path to the trained .keras model file (default: {DEFAULT_MODEL_PATH})"
    )
    args = parser.parse_args()

    # Load model
    try:
        model = load_cnn_model(args.model_path)
    except Exception as e:
        print(f"[Fatal Error] {e}")
        sys.exit(1)

    # Determine images to test
    if args.image_path is not None:
        predict_image(args.image_path, model)
    else:
        print("\nNo image path specified. Demonstrating predictions on images in CNN/test_images/:\n")
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        test_images = sorted([
            os.path.join(DEFAULT_TEST_DIR, f)
            for f in os.listdir(DEFAULT_TEST_DIR)
            if os.path.splitext(f.lower())[1] in valid_exts
        ]) if os.path.exists(DEFAULT_TEST_DIR) else []

        if not test_images:
            print(f"No sample images found in '{DEFAULT_TEST_DIR}'.")
            print("Usage: python predict.py <path_to_image>")
            sys.exit(1)

        for img_path in test_images:
            predict_image(img_path, model)

        print("\nTo predict a specific image, run:")
        print("  python predict.py path/to/your_image.jpg")


if __name__ == "__main__":
    main()
