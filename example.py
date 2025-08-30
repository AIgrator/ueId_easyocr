# This script tests the fully integrated UIED framework with EasyOCR.
# It is a self-contained, interactive example.

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
from uied_engine.detect_compo.ip_region_proposal import detect_components, classify_components
from uied_engine.detect_merge.merge import merge
from classifier import TestClassifier as CNN
from uied_engine.config.CONFIG_UIED import Config as UIEDConfig
from uied_engine.detect_compo.lib_ip import file_utils as file
from uied_engine.detect_compo.lib_ip import ip_draw as draw

# --- SCREENSHOT FUNCTIONALITY (LOCALIZED) ---
def get_screenshot() -> Image.Image:
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
    merge_dir = pjoin(temp_dir, 'merge')
    classifier_dir = pjoin(temp_dir, 'classifier')
    os.makedirs(ip_dir, exist_ok=True)
    os.makedirs(ocr_dir, exist_ok=True)
    os.makedirs(merge_dir, exist_ok=True)
    os.makedirs(classifier_dir, exist_ok=True)

    name = os.path.basename(screenshot_path).replace('.png', '')
    raw_compo_json_path = pjoin(ip_dir, name + '.json')
    ocr_json_path = pjoin(ocr_dir, name + '.json')
    merge_json_path = pjoin(merge_dir, name + '.json')
    classified_json_path = pjoin(classifier_dir, name + '.json')
    classified_img_path = pjoin(classifier_dir, name + '.jpg')

    try:
        cpu_readings = []
        psutil.cpu_percent(interval=None)
        start_time = time.time()

        text_detection(screenshot_path, ocr_json_path, ocr_reader=ocr_reader, show=False)
        cpu_readings.append(psutil.cpu_percent(interval=None))
        
        org_img, _, raw_components = detect_components(screenshot_path, uied_params=key_params, resize_by_height=800)
        file.save_corners_json(raw_compo_json_path, raw_components)
        cpu_readings.append(psutil.cpu_percent(interval=None))

        classifier = {'Elements': CNN()}
        classified_components = classify_components(org_img, raw_components, classifier)
        file.save_corners_json(classified_json_path, classified_components)
        draw.draw_bounding_box_class(org_img, classified_components, write_path=classified_img_path)
        cpu_readings.append(psutil.cpu_percent(interval=None))
        
        if not os.path.exists(classified_json_path) or not os.path.exists(ocr_json_path):
             logging.error("Detection files (classified.json or ocr.json) not found. Halting merge.")
             return False
        merge(screenshot_path, classified_json_path, ocr_json_path, merge_json_path,
              is_remove_bar=key_params['remove-bar'],
              is_paragraph=key_params['merge-line-to-paragraph'], show=False)
        cpu_readings.append(psutil.cpu_percent(interval=None))
        
        end_time = time.time()
        if cpu_readings:
            max_cpu = max(cpu_readings)
            avg_cpu = sum(cpu_readings) / len(cpu_readings)
            print("\n--- Performance Report ---")
            print(f"Total processing time: {end_time - start_time:.2f} seconds")
            print(f"Maximum CPU Load: {max_cpu:.2f}%")
            print(f"Average CPU Load: {avg_cpu:.2f}%")
        
        logging.info("UIED pipeline executed successfully.")
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

    print("\n--- UIED Interactive Test ---")
    print("Press 'F3' to capture the screen and run analysis.")
    print("Press 'Esc' to exit.")

    while True:
        event = keyboard.read_event(suppress=True)

        if event.event_type == keyboard.KEY_DOWN:
            if event.name == 'f3':
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                logging.info(f"F3 pressed. Starting UIED integration test (Timestamp: {timestamp})...")

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
                    result_image_path = os.path.join(temp_dir, "merge", "screenshot.jpg")
                    if os.path.exists(result_image_path):
                        final_image = cv2.imread(result_image_path)
                        final_filename = os.path.join(screenshots_dir, f"uied_final_detection_{timestamp}.png")
                        cv2.imwrite(final_filename, final_image)
                        print(f"Detection finished successfully! Review the visual result: {final_filename}")
                    else:
                        print("Pipeline reported success, but the final merged image was not found.")

                    classification_viz_path = os.path.join(temp_dir, "classifier", "screenshot.jpg")
                    if os.path.exists(classification_viz_path):
                        final_clf_filename = os.path.join(screenshots_dir, f"uied_classification_only_{timestamp}.png")
                        shutil.copy(classification_viz_path, final_clf_filename)
                        print(f"Review the classification-only result: {final_clf_filename}")
                    else:
                        print("The classification-only visualization was not found.")
                else:
                    print("\n--- UIED Pipeline Failed ---")
                    print("Please check the logs for errors.")

                print("\nReady for next capture. Press 'F3' or 'Esc' to exit.")

            elif event.name == 'esc':
                print("\nExiting...")
                break

if __name__ == "__main__":
    main()