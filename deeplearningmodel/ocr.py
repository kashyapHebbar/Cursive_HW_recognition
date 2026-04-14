"""Reusable OCR helpers for character segmentation and inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import imutils
import numpy as np
from imutils.contours import sort_contours


@dataclass
class CharacterPrediction:
    label: str
    probability: float
    bbox: Tuple[int, int, int, int]


def get_label_names() -> List[str]:
    return list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def extract_character_rois(gray_image: np.ndarray) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
    """Extract normalized 32x32 character ROIs and their bounding boxes."""
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    contours = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if not contours:
        return []

    contours = sort_contours(contours, method="left-to-right")[0]
    chars: List[Tuple[np.ndarray, Tuple[int, int, int, int]]] = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if not ((5 <= w <= 150) and (15 <= h <= 120)):
            continue

        roi = gray_image[y : y + h, x : x + w]
        thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        (t_h, t_w) = thresh.shape

        if t_w > t_h:
            thresh = imutils.resize(thresh, width=32)
        else:
            thresh = imutils.resize(thresh, height=32)

        (t_h, t_w) = thresh.shape
        d_x = int(max(0, 32 - t_w) / 2.0)
        d_y = int(max(0, 32 - t_h) / 2.0)

        padded = cv2.copyMakeBorder(
            thresh,
            top=d_y,
            bottom=d_y,
            left=d_x,
            right=d_x,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0),
        )
        padded = cv2.resize(padded, (32, 32)).astype("float32") / 255.0
        padded = np.expand_dims(padded, axis=-1)
        chars.append((padded, (x, y, w, h)))

    return chars


def predict_characters(model, image: np.ndarray, min_confidence: float = 0.0) -> Tuple[str, List[CharacterPrediction], np.ndarray]:
    """Predict characters in an input image and return text + annotations."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    extracted = extract_character_rois(gray)

    if not extracted:
        return "", [], image.copy()

    boxes = [item[1] for item in extracted]
    chars = np.array([item[0] for item in extracted], dtype="float32")
    probs = model.predict(chars, verbose=0)
    labels = get_label_names()

    annotated = image.copy()
    predictions: List[CharacterPrediction] = []

    for pred, (x, y, w, h) in zip(probs, boxes):
        idx = int(np.argmax(pred))
        probability = float(pred[idx])
        label = labels[idx]

        if probability < min_confidence:
            continue

        predictions.append(CharacterPrediction(label=label, probability=probability, bbox=(x, y, w, h)))
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )

    text = "".join(item.label for item in predictions)
    return text, predictions, annotated
