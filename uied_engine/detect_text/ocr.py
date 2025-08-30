"""
This module is a modified version of the original ocr.py from UIED.
It replaces the dependency on the Google Vision API with our existing
EasyOCR reader to perform text detection locally and for free.
"""
import os
import json
import cv2
import numpy as np
from os.path import join as pjoin

def bounds_to_poly(bounds):
    """
    Converts EasyOCR bounds (a list of 4 points) into the
    boundingPoly format required by the UIED merge module.
    """
    # EasyOCR format: [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    # Target format: {'vertices': [{'x': x1, 'y': y1}, ...]}
    vertices = []
    for point in bounds:
        vertices.append({'x': int(point[0]), 'y': int(point[1])})
    return {'vertices': vertices}

def recognize_text_easyocr(img_path, ocr_reader):
    """
    Performs OCR using the provided EasyOCR reader instance and
    formats the output to be compatible with the UIED framework.

    Args:
        img_path (str): The path to the image file.
        ocr_reader (easyocr.Reader): An initialized EasyOCR reader instance.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              a detected text block in a format similar to Google OCR's output.
              Returns an empty list if OCR fails or no text is found.
    """
    if ocr_reader is None:
        print("ERROR: EasyOCR reader is not provided.")
        return []
    
    try:
        # readtext returns a list of (bounding_box, text, confidence)
        results = ocr_reader.readtext(img_path)
        
        if not results:
            return []

        # Convert results to the format expected by the merge module
        formatted_results = []
        for (bbox, text, _) in results:
            formatted_results.append({
                "description": text,
                "boundingPoly": bounds_to_poly(bbox)
            })
        
        return formatted_results

    except Exception as e:
        print(f"An error occurred during EasyOCR processing: {e}")
        return []

def ocr_detection_easyocr(img_path, ocr_reader, output_json_path):
    """
    A wrapper function that performs OCR and saves the results to a JSON file.
    This function replaces the original Google-based ocr_detection.
    """
    # Perform OCR
    ocr_result = recognize_text_easyocr(img_path, ocr_reader)
    
    # Ensure the output directory exists before writing the file
    output_dir = os.path.dirname(output_json_path)
    os.makedirs(output_dir, exist_ok=True)

    # Save the result to the specified JSON file path
    with open(output_json_path, 'w') as f:
        json.dump(ocr_result, f, indent=4)
    
    # The original function returned the image, but it's not used.
    # We'll just print a confirmation.
    # print(f"EasyOCR detection completed. Results saved to {json_path}")


# Note: The original google_ocr_detection function is left here for reference
# but is no longer used.

def google_ocr_detection(img_path, ocr_root):
    """
    DEPRECATED: This is the original function that uses Google Cloud Vision API.
    It is replaced by ocr_detection_easyocr.
    """
    raise NotImplementedError("Google OCR has been replaced by EasyOCR. Do not call this function.")