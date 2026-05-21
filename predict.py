import argparse
import json
import os
import sys

import numpy as np
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image_bytes
import logging
from exceptions import PreprocessingError, ModelExecutionError
from model_utils import ensure_model_file, get_model_path, get_model_url, get_model_sha256

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loading (lazy singleton)
# ---------------------------------------------------------------------------

_model = None


def load_deepfake_model(model_path: str | None = None):
    """Load (or return cached) deepfake-detection model.

    Parameters
    ----------
    model_path:
        Override the model file location.  When *None*, the path is resolved
        via ``model_utils.get_model_path()`` / ``PIXELTRUTH_MODEL_PATH``.
    """
    global _model
    if _model is not None:
        return _model

    resolved_path = model_path or get_model_path()
    model_file = ensure_model_file(
        model_path=resolved_path,
        model_url=get_model_url(),
        model_sha256=get_model_sha256(),
        download_if_missing=True,
    )
    _model = load_model(model_file)
    return _model


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}


def preprocess_image(image_path: str) -> np.ndarray:
    """Read and preprocess an image for the model via shared byte-based pipeline.

    Parameters
    ----------
    image_path:
        Filesystem path to the image file.

    Returns
    -------
    np.ndarray
        Shape ``(1, 96, 96, 3)``, values in ``[0, 1]``.

    Raises
    ------
    FileNotFoundError
        When *image_path* does not exist on disk.
    ValueError
        When the file extension is not supported.
    PreprocessingError
        When the file exists but cannot be decoded or preprocessed.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    try:
        with open(image_path, "rb") as file_handle:
            image_bytes = file_handle.read()
        return preprocess_image_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Image preprocessing failed for {image_path}: {e}", exc_info=True)
        raise PreprocessingError(f"Failed to preprocess image: {str(e)}") from e


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_image(image_path: str, model_path: str | None = None) -> dict:
    """Run deepfake detection on a single image.

    Parameters
    ----------
    image_path:
        Path to the image to classify.
    model_path:
        Optional override for the model file location.

    Returns
    -------
    dict
        ``{"image": str, "label": "Real"|"Fake", "confidence": float,
           "raw": list[float]}``

    Raises
    ------
    FileNotFoundError
        When *image_path* does not exist on disk.
    ValueError
        When the file extension is not supported.
    PreprocessingError
        When the image cannot be decoded or preprocessed.
    ModelExecutionError
        When model inference fails.
    """
    image = preprocess_image(image_path)

    try:
        model = load_deepfake_model(model_path)
        prediction = model.predict(image, verbose=0)
    except (PreprocessingError, FileNotFoundError, ValueError):
        raise
    except Exception as e:
        logger.error(f"Model prediction failed: {e}", exc_info=True)
        raise ModelExecutionError(f"Model prediction failed: {str(e)}") from e

    class_index = int(np.argmax(prediction, axis=1)[0])
    confidence = float(np.max(prediction)) * 100
    # Dataset mapping: class 0 = Real, class 1 = Fake
    label = "Fake" if class_index == 1 else "Real"

    return {
        "image": image_path,
        "label": label,
        "confidence": round(confidence, 1),
        "raw": prediction[0].tolist(),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="predict.py",
        description=(
            "PixelTruth — deepfake image detector.\n"
            "Classifies one or more images as Real or Fake."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python predict.py photo.jpg\n"
            "  python predict.py img1.jpg img2.png --json\n"
            "  python predict.py --model /weights/model.h5 photo.jpg\n\n"
            "Environment variables:\n"
            "  PIXELTRUTH_MODEL_PATH   path to model file\n"
            "  PIXELTRUTH_MODEL_URL    URL to download model if missing\n"
            "  PIXELTRUTH_MODEL_SHA256 expected SHA-256 of the model file"
        ),
    )
    parser.add_argument(
        "images",
        metavar="IMAGE",
        nargs="+",
        help="path(s) to image file(s) to classify",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        default=None,
        help=(
            "path to the .h5 model file "
            "(default: $PIXELTRUTH_MODEL_PATH or 'deepfake_detection_model.h5')"
        ),
    )
    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="print results as JSON (useful for scripting)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress informational messages; only print results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point.  Returns 0 on success, 1 if any image fails."""
    parser = build_parser()
    args = parser.parse_args(argv)

    results = []
    exit_code = 0

    for image_path in args.images:
        try:
            result = predict_image(image_path, model_path=args.model)
            results.append(result)
        except (FileNotFoundError, ValueError, PreprocessingError, ModelExecutionError) as exc:
            # Non-fatal: report the error and continue with remaining images.
            error_result = {
                "image": image_path,
                "error": str(exc),
            }
            results.append(error_result)
            exit_code = 1
            if not args.quiet:
                print(f"[ERROR] {exc}", file=sys.stderr)

    if args.output_json:
        print(json.dumps(results if len(results) > 1 else results[0], indent=2))
    else:
        for result in results:
            if "error" in result:
                continue  # already printed to stderr above
            if not args.quiet:
                print(f"\nImage      : {result['image']}")
                print(f"Raw output : {result['raw']}")
            print(f"Prediction : {result['label']}")
            print(f"Confidence : {result['confidence']:.1f}%")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())