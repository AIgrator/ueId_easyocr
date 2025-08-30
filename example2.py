# This script tests a modified UIED pipeline that excludes text components.
# It detects UI components and filters out any that overlap significantly with OCR-detected text.

import os
import logging
import time
from datetime import datetime
import cv2
import json
from os.path import join as pjoin
import psutil
import mss
from PIL import Image
import keyboard
import numpy as np
import shutil

# --- CORRECT IMPORTS ---
from logging_config import setup_logging
from uied_engine.detect_text.text_detection import text_detection
from uied_engine.detect_compo.ip_region_proposal import detect_components
from uied_engine.detect_compo.lib_ip import file_utils as file
from uied_engine.detect_compo.lib_ip import ip_draw as draw

# --- UTILITY FUNCTIONS ---

def calculate_iou(box_a, box_b):
    """
    Calculate the ratio of the intersection area over the area of the first box (box_a).
    Each box is in the format [x_min, y_min, x_max, y_max].
    This helps determine how much of box_a is contained within box_b.
    """
    # Determine the coordinates of the intersection rectangle
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    # Compute the area of intersection
    inter_area = max(0, x_b - x_a) * max(0, y_b - y_a)

    # Compute the area of the first bounding box
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    
    if box_a_area == 0:
        return 0.0
        
    # Compute the ratio of intersection over the area of the first box (component)
    ratio = inter_area / float(box_a_area)

    return ratio

def get_screenshot() -> Image.Image:
    """Captures the primary monitor screenshot."""
    with mss.mss() as sct:
        sct_img = sct.grab(sct.monitors[1])
        img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
        return img

# --- MAIN PIPELINE ---
def run_uied_pipeline(screenshot_path, temp_dir, ocr_reader):
    if ocr_reader is None:
        logging.error("OCR reader not initialized.")
        return False

    key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':50,
                  'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}
    
    ip_dir = pjoin(temp_dir, 'ip')
    ocr_dir = pjoin(temp_dir, 'ocr')
    os.makedirs(ip_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)

    name = os.path.basename(screenshot_path).replace('.png', '')
    raw_compo_json_path = pjoin(ip_dir, name + '.json')
    ip_img_path = pjoin(ip_dir, name + '.jpg')
    ocr_json_path = pjoin(ocr_dir, name + '.json')

    try:
        cpu_readings = []
        psutil.cpu_percent(interval=None)
        start_time = time.time()

        # 1. Detect Text
        text_detection(screenshot_path, ocr_json_path, ocr_reader=ocr_reader, show=False)
        cpu_readings.append(psutil.cpu_percent(interval=None))
        
        # 2. Detect all potential components
        org_img, _, raw_components = detect_components(screenshot_path, uied_params=key_params, resize_by_height=800)
        cpu_readings.append(psutil.cpu_percent(interval=None))

        # 3. Filter out components that are likely text
        if not os.path.exists(ocr_json_path):
            logging.error("OCR results not found. Cannot filter text components.")
            return False
        
        with open(ocr_json_path, 'r') as f:
            ocr_data = json.load(f)

        # Get original and resized dimensions to calculate scaling factor
        original_shape = ocr_data['img_shape'] # [height, width, channels]
        resized_shape = org_img.shape          # [height, width, channels]

        height_ratio = resized_shape[0] / original_shape[0]
        width_ratio = resized_shape[1] / original_shape[1]
        
        text_boxes = []
        for item in ocr_data['texts']:
            # Scale the coordinates from the original image size to the resized image size
            x_min = int(item['column_min'] * width_ratio)
            y_min = int(item['row_min'] * height_ratio)
            x_max = int(item['column_max'] * width_ratio)
            y_max = int(item['row_max'] * height_ratio)
            text_boxes.append([x_min, y_min, x_max, y_max])

        non_text_components = []
        overlap_threshold = 0.9  # 90% overlap

        for compo in raw_components:
            is_text = False
            compo_box = compo.put_bbox() # [col_min, row_min, col_max, row_max]
            
            for text_box in text_boxes:
                # If 90% of the component's area is inside a text box, it's considered text
                if calculate_iou(compo_box, text_box) > overlap_threshold:
                    is_text = True
                    break
            
            if not is_text:
                non_text_components.append(compo)
        
        logging.info(f"Component Filtering (Text): Original: {len(raw_components)}, Non-Text: {len(non_text_components)}")

        # 4. Filter components by size and aspect ratio to find icon/button-like elements
        filtered_components = []
        resized_height, resized_width = org_img.shape[:2]

        # Relative size constraints based on the resized image (height=800px)
        # Height: 8px to 32px -> 1% to 4% of 800px
        # Width: up to 200px -> 25% of 800px
        min_h_rel = 0.01  # Corresponds to 8px on an 800px tall image
        max_h_rel = 0.04  # Corresponds to 32px on an 800px tall image
        max_w_rel = 0.25  # Corresponds to 200px on an 800px tall image
        
        # Aspect ratio constraints to favor squares and horizontal rectangles
        min_aspect_ratio = 0.5 # width/height
        max_aspect_ratio = 5.0 # width/height

        for compo in non_text_components:
            width = compo.width
            height = compo.height

            if height == 0: continue

            aspect_ratio = width / float(height)
            
            # Apply filters based on relative size and aspect ratio
            is_right_height = (min_h_rel * resized_height) <= height <= (max_h_rel * resized_height)
            is_right_width = width <= (max_w_rel * resized_width)
            is_right_aspect_ratio = min_aspect_ratio <= aspect_ratio <= max_aspect_ratio

            if is_right_height and is_right_width and is_right_aspect_ratio:
                filtered_components.append(compo)

        logging.info(f"Component Filtering (Size/Shape): Non-Text: {len(non_text_components)}, Filtered: {len(filtered_components)}")

        # 5. Save the final filtered components
        file.save_corners_json(raw_compo_json_path, filtered_components)
        draw.draw_bounding_box(org_img, filtered_components, write_path=ip_img_path, show=False)
        cpu_readings.append(psutil.cpu_percent(interval=None))
        
        end_time = time.time()
        if cpu_readings:
            max_cpu = max(cpu_readings)
            avg_cpu = sum(cpu_readings) / len(cpu_readings)
            print("\n--- Performance Report ---")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            print(f"Maximum CPU Load: {max_cpu:.2f}%")
            print(f"Average CPU Load: {avg_cpu:.2f}%")
        
        logging.info("UIED non-text component detection executed successfully.")
        return True

    except Exception as e:
        logging.error(f"An error occurred during the UIED pipeline execution: {e}", exc_info=True)
        return False

# --- MAIN EXECUTION BLOCK ---
def main():
    setup_logging()
    
    print("Initializing EasyOCR Reader...")
    try:
        import easyocr
        ocr_reader = easyocr.Reader(['en', 'ru'])
        print("EasyOCR Reader initialized successfully.")
    except Exception as ocr_error:
        logging.error(f"Fatal: Failed to initialize EasyOCR: {ocr_error}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    temp_dir = os.path.join(script_dir, "temp_output")
    screenshots_dir = os.path.join(script_dir, "screenshots")
    os.makedirs(temp_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    print("\n--- UIED Non-Text Component Detection ---")
    print("Press 'F3' to capture the screen and run analysis.")
    print("Press 'Esc' to exit.")

    while True:
        event = keyboard.read_event(suppress=True)

        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'f3':
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                logging.info(f"F3 pressed. Starting UIED non-text detection (Timestamp: {timestamp})...")

                print("\nCapturing screenshot...")
                screenshot_pil = get_screenshot()
                if not screenshot_pil:
                    logging.error("Failed to capture screenshot.")
                    continue
                
                temp_screenshot_path = os.path.join(temp_dir, "screenshot.png")
                screenshot_pil.save(temp_screenshot_path)
                logging.info(f"Screenshot captured: {temp_screenshot_path}")

                success = run_uied_pipeline(temp_screenshot_path, temp_dir, ocr_reader)

                if success:
                    print("\n--- Results ---")
                    result_image_path = os.path.join(temp_dir, "ip", "screenshot.jpg")
                    if os.path.exists(result_image_path):
                        final_image = cv2.imread(result_image_path)
                        final_filename = os.path.join(screenshots_dir, f"uied_non_text_{timestamp}.png")
                        cv2.imwrite(final_filename, final_image)
                        print(f"Detection finished successfully! Review the visual result: {final_filename}")
                    else:
                        print("Pipeline reported success, but the final result image was not found.")
                else:
                    print("\n--- UIED Pipeline Failed ---")
                    print("Please check the logs for errors.")

                print("\nReady for next capture. Press 'F3' or 'Esc' to exit.")

            elif event.name == 'esc':
                print("\nExiting...")
                break

if __name__ == "__main__":
    main()
