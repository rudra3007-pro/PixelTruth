"""
Unit tests for preprocess_image and predict_image in app.py.

Run with:
    pytest test_pipeline.py -v

These tests do NOT require the real model file. predict_image tests use a
mock model so the suite runs in CI without any large file downloads.
"""

import types
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Inline copies of the functions under test (from app.py)
# This avoids importing app.py, which triggers Streamlit globals on import.
# ---------------------------------------------------------------------------

from tensorflow.keras.preprocessing.image import img_to_array


def preprocess_image(image):
    image = cv2.resize(image, (96, 96))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image


def predict_image(image, model):
    """Thin wrapper matching app.py logic, with model passed explicitly for testability."""
    if model is None:
        return None, None
    processed = preprocess_image(image)
    prediction = model.predict(processed, verbose=0)
    class_label = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction))
    label = "Real" if class_label == 0 else "Fake"
    return label, confidence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_blank_image(h=200, w=200, channels=3):
    """Return a solid-colour BGR image as a numpy array."""
    return np.full((h, w, channels), 128, dtype=np.uint8)


def make_mock_model(prediction_array):
    """Return a mock Keras model whose .predict() returns prediction_array."""
    mock = MagicMock()
    mock.predict.return_value = prediction_array
    return mock


# ---------------------------------------------------------------------------
# preprocess_image tests
# ---------------------------------------------------------------------------

class TestPreprocessImage:

    def test_output_shape_is_1_96_96_3(self):
        """Output tensor must always be (1, 96, 96, 3) regardless of input size."""
        for size in [(50, 50), (200, 200), (1024, 768)]:
            image = make_blank_image(*size)
            result = preprocess_image(image)
            assert result.shape == (1, 96, 96, 3), (
                f"Expected (1, 96, 96, 3) for input size {size}, got {result.shape}"
            )

    def test_pixel_values_normalised_between_0_and_1(self):
        """All pixel values must be in [0.0, 1.0] after preprocessing."""
        image = make_blank_image()
        result = preprocess_image(image)
        assert result.min() >= 0.0, "Pixel values dropped below 0.0"
        assert result.max() <= 1.0, "Pixel values exceeded 1.0"

    def test_all_white_image_normalises_to_1(self):
        """A pure-white image (255, 255, 255) should normalise to 1.0."""
        image = np.full((100, 100, 3), 255, dtype=np.uint8)
        result = preprocess_image(image)
        assert np.allclose(result, 1.0), "White image did not normalise to 1.0"

    def test_all_black_image_normalises_to_0(self):
        """A pure-black image (0, 0, 0) should normalise to 0.0."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        result = preprocess_image(image)
        assert np.allclose(result, 0.0), "Black image did not normalise to 0.0"

    def test_output_dtype_is_float(self):
        """Output must be a float array, not uint8."""
        image = make_blank_image()
        result = preprocess_image(image)
        assert np.issubdtype(result.dtype, np.floating), (
            f"Expected float dtype, got {result.dtype}"
        )

    def test_non_square_input_resized_correctly(self):
        """A non-square image must still produce (1, 96, 96, 3)."""
        image = make_blank_image(h=480, w=640)
        result = preprocess_image(image)
        assert result.shape == (1, 96, 96, 3)

    def test_none_input_raises(self):
        """Passing None should raise an error, not return silently wrong data."""
        with pytest.raises(Exception):
            preprocess_image(None)


# ---------------------------------------------------------------------------
# predict_image tests
# ---------------------------------------------------------------------------

class TestPredictImage:

    def test_returns_real_label_when_class_0_wins(self):
        """When model predicts class 0 with high confidence, label must be 'Real'."""
        model = make_mock_model(np.array([[0.9, 0.1]]))
        image = make_blank_image()
        label, confidence = predict_image(image, model)
        assert label == "Real"

    def test_returns_fake_label_when_class_1_wins(self):
        """When model predicts class 1 with high confidence, label must be 'Fake'."""
        model = make_mock_model(np.array([[0.1, 0.9]]))
        image = make_blank_image()
        label, confidence = predict_image(image, model)
        assert label == "Fake"

    def test_confidence_is_float(self):
        """Confidence score must be a Python float."""
        model = make_mock_model(np.array([[0.8, 0.2]]))
        image = make_blank_image()
        label, confidence = predict_image(image, model)
        assert isinstance(confidence, float), (
            f"Expected float confidence, got {type(confidence)}"
        )

    def test_confidence_between_0_and_1(self):
        """Confidence must be in [0.0, 1.0]."""
        model = make_mock_model(np.array([[0.7, 0.3]]))
        image = make_blank_image()
        label, confidence = predict_image(image, model)
        assert 0.0 <= confidence <= 1.0, f"Confidence out of range: {confidence}"

    def test_none_model_returns_none_none(self):
        """If no model is loaded, both return values must be None."""
        image = make_blank_image()
        label, confidence = predict_image(image, model=None)
        assert label is None
        assert confidence is None

    def test_model_receives_preprocessed_input(self):
        """model.predict must be called with shape (1, 96, 96, 3)."""
        model = make_mock_model(np.array([[0.6, 0.4]]))
        image = make_blank_image(h=300, w=400)
        predict_image(image, model)
        call_args = model.predict.call_args[0][0]
        assert call_args.shape == (1, 96, 96, 3), (
            f"model.predict received wrong shape: {call_args.shape}"
        )

    def test_label_is_one_of_valid_classes(self):
        """Label must be exactly 'Real' or 'Fake', nothing else."""
        for prediction in [np.array([[0.9, 0.1]]), np.array([[0.1, 0.9]])]:
            model = make_mock_model(prediction)
            label, _ = predict_image(make_blank_image(), model)
            assert label in ("Real", "Fake"), f"Unexpected label: {label}"