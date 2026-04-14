import numpy as np

from deeplearningmodel.ocr import get_label_names, predict_characters


class DummyModel:
    def predict(self, batch, verbose=0):
        result = np.zeros((batch.shape[0], 36), dtype="float32")
        result[:, 10] = 1.0  # 'A'
        return result


def test_label_names_size_and_order():
    labels = get_label_names()
    assert len(labels) == 36
    assert labels[0] == "0"
    assert labels[10] == "A"
    assert labels[-1] == "Z"


def test_predict_characters_no_contours_returns_empty():
    image = np.zeros((64, 64, 3), dtype="uint8")
    text, predictions, annotated = predict_characters(DummyModel(), image)

    assert text == ""
    assert predictions == []
    assert annotated.shape == image.shape
