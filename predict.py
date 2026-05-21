import argparse
import json
import os
import sys
import logging

import numpy as np

from preprocessing import preprocess_image_bytes
from exceptions import (
    PreprocessingError,
    ModelExecutionError,
)

from utils.model_loader import load_cached_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tiff",
    ".tif",
}


def preprocess_image(image_path: str) -> np.ndarray:
    """Read and preprocess an image."""

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

        logger.error(
            f"Image preprocessing failed for {image_path}: {e}",
            exc_info=True
        )

        raise PreprocessingError(
            f"Failed to preprocess image: {str(e)}"
        ) from e


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_image(image_path: str) -> dict:
    """Run deepfake detection on a single image."""

    image = preprocess_image(image_path)

    try:

        # Cached lazy-loaded model
        model = load_cached_model()

        prediction = model.predict(image, verbose=0)

    except (
        PreprocessingError,
        FileNotFoundError,
        ValueError,
    ):
        raise

    except Exception as e:

        logger.error(
            f"Model prediction failed: {e}",
            exc_info=True
        )

        raise ModelExecutionError(
            f"Model prediction failed: {str(e)}"
        ) from e

    class_index = int(np.argmax(prediction, axis=1)[0])

    confidence = float(np.max(prediction)) * 100

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
    )

    parser.add_argument(
        "images",
        metavar="IMAGE",
        nargs="+",
        help="path(s) to image file(s) to classify",
    )

    parser.add_argument(
        "--json",
        dest="output_json",
        action="store_true",
        help="print results as JSON",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="suppress informational messages",
    )

    return parser


def main(argv: list[str] | None = None) -> int:

    parser = build_parser()

    args = parser.parse_args(argv)

    results = []

    exit_code = 0

    for image_path in args.images:

        try:

            result = predict_image(image_path)

            results.append(result)

        except (
            FileNotFoundError,
            ValueError,
            PreprocessingError,
            ModelExecutionError,
        ) as exc:

            error_result = {
                "image": image_path,
                "error": str(exc),
            }

            results.append(error_result)

            exit_code = 1

            if not args.quiet:
                print(f"[ERROR] {exc}", file=sys.stderr)

    if args.output_json:

        print(
            json.dumps(
                results if len(results) > 1 else results[0],
                indent=2
            )
        )

    else:

        for result in results:

            if "error" in result:
                continue

            if not args.quiet:
                print(f"\nImage      : {result['image']}")
                print(f"Raw output : {result['raw']}")

            print(f"Prediction : {result['label']}")
            print(f"Confidence : {result['confidence']:.1f}%")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())