class PixelTruthError(Exception):
    """Base exception for all PixelTruth related errors."""
    pass

class PreprocessingError(PixelTruthError):
    """Raised when an error occurs during image preprocessing (resizing, format conversion, etc.)."""
    pass

class ModelExecutionError(PixelTruthError):
    """Raised when the deep learning model encounters an error during inference."""
    pass

class ModelDownloadError(PixelTruthError):
    """Raised when the model file cannot be downloaded (network error, HTTP error, timeout, etc.)."""
    pass
