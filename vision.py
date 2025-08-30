"""
This module is responsible for understanding what is on the screen.
It handles screen capture and OCR (Optical Character Recognition).
"""
import time
import io
import logging
import numpy as np
from PySide6.QtWidgets import QWidget
import mss
from PIL import Image
import easyocr

_settings = {}
_ocr_reader = None

def _initialize_ocr_reader():
    """Initializes the EasyOCR reader if it hasn't been already."""
    global _ocr_reader
    if _ocr_reader is None:
        logging.info("Initializing EasyOCR Reader for ['en', 'ru']...")
        try:
            _ocr_reader = easyocr.Reader(['en', 'ru'])
            logging.info("EasyOCR Reader initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize EasyOCR Reader: {e}")
            # To prevent re-initialization attempts on every call after a failure
            _ocr_reader = "initialization_failed" 

def perform_ocr(image: Image.Image) -> list:
    """
    Performs OCR on the given PIL Image and returns the results.

    Args:
        image (Image.Image): The image to process.

    Returns:
        list: A list of tuples, where each tuple contains the bounding box,
              recognized text, and confidence score.
    """
    _initialize_ocr_reader()
    if not isinstance(_ocr_reader, easyocr.Reader):
        logging.error("OCR Reader is not available. Cannot perform OCR.")
        return []
    
    try:
        image_np = np.array(image)
        results = _ocr_reader.readtext(image_np)
        return results
    except Exception as e:
        logging.error(f"An error occurred during OCR processing: {e}")
        return []

def update_settings(new_settings: dict):
    """Updates the settings for the vision module."""
    global _settings
    _settings = new_settings
    logging.info(f"Vision module settings updated. WebP quality: {_settings.get('webp_quality')}")

def get_screenshot() -> Image.Image:
    """
    Captures the entire screen and returns it as a Pillow Image object.

    Returns:
        Image.Image: The captured screenshot.
    """
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        return img

def get_clean_screenshot(window: QWidget) -> Image.Image:
    """
    Hides the specified window, takes a screenshot, and returns it as a PIL Image.

    Args:
        window (QWidget): The window to hide and show.

    Returns:
        Image.Image: The screenshot as a PIL Image object, or None on failure.
    """
    try:
        window.hide()
        time.sleep(0.3) 
        screenshot = get_screenshot()
        return screenshot
    except Exception as e:
        logging.error(f"Failed to get a clean screenshot: {e}")
        return None
    finally:
        window.show()

def compress_image_to_bytes(image: Image.Image) -> bytes:
    """
    Compresses a PIL Image into WebP format in memory, using settings.

    Args:
        image (Image.Image): The image to compress.

    Returns:
        bytes: The compressed screenshot in WebP format.
    """
    if not image:
        return None
    try:
        # Get compression settings from the global settings dict
        use_lossless = _settings.get("webp_lossless", True)
        quality = _settings.get("webp_quality", 80)
        
        logging.info(f"Compressing image with WebP. Lossless: {use_lossless}, Quality: {quality if not use_lossless else 'N/A'}")

        buffer = io.BytesIO()
        image.save(buffer, "webp", lossless=use_lossless, quality=quality)
        return buffer.getvalue()
    except Exception as e:
        logging.error(f"Failed to compress image: {e}")
        return None

