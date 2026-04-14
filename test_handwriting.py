# USAGE
# python test_handwriting.py --model handwriting.model --image images/myimage.jpeg

import argparse

import cv2
from tensorflow.keras.models import load_model

from deeplearningmodel.ocr import predict_characters


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-m", "--model", type=str, required=True, help="path to trained handwriting recognition model")
ap.add_argument("--min-confidence", type=float, default=0.0, help="minimum probability threshold [0,1]")
ap.add_argument("--no-display", action="store_true", help="disable OpenCV image preview")
args = vars(ap.parse_args())

print("[INFO] loading handwriting OCR model...")
model = load_model(args["model"])

image = cv2.imread(args["image"])
if image is None:
    raise ValueError(f"Unable to read image: {args['image']}")

text, predictions, annotated = predict_characters(model, image, min_confidence=args["min_confidence"])

if not predictions:
    print("[WARN] No characters were detected.")
else:
    for prediction in predictions:
        print(f"[INFO] {prediction.label} - {prediction.probability * 100:.2f}% @ {prediction.bbox}")

print(f"[INFO] text: {text}")

if not args["no_display"]:
    cv2.imshow("Image", annotated)
    cv2.waitKey(0)
