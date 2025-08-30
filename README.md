# UIED-EasyOCR Integration

This repository contains scripts for integrating the UIED (UI Element Detection) framework with EasyOCR for robust text and component recognition from screenshots. The primary goal is to provide a toolkit for analyzing UI structures from images.

The core technology combines computer vision techniques to identify graphical UI elements and an OCR engine to extract textual information.

## Getting Started

### 1. Download the Models

The classification models are not stored in this repository. You need to download them manually.

-   **Download Link:** [Google Drive](https://drive.google.com/drive/folders/1wvzOaEClgKdnTOmnWELRXr9kyeSVKuxQ?usp=drive_link)
-   **Installation:** Place the downloaded `.h5` files into the `models/` directory at the root of this project.

## Example Scripts

This repository includes three interactive example scripts that demonstrate different capabilities of the integration. To use any of them, run the script and press **F3** to capture the screen and initiate the analysis. Press **Esc** to exit.

### `example.py`: Full UIED Pipeline

This script showcases the complete, integrated UIED pipeline. It performs the following steps:
1.  **Captures a screenshot.**
2.  **Detects UI components** using the `detect_components` function.
3.  **Recognizes text** using EasyOCR via the `text_detection` function.
4.  **Classifies** the detected non-text components (e.g., buttons, icons) using a CNN classifier.
5.  **Merges** the results from component detection, classification, and OCR.
6.  **Saves the final annotated image**, showing bounding boxes around all detected UI elements and text.

This is the main example for getting a comprehensive analysis of a user interface.

### `example1.py`: Non-Text Component Detection

This script demonstrates how to isolate graphical, non-text UI elements. It's useful when you only care about the layout of interactive elements and want to ignore text.
1.  **Captures a screenshot.**
2.  **Detects all potential UI components.**
3.  **Detects all text blocks** using OCR.
4.  **Filters out components** that have a high degree of overlap (90% or more) with the detected text blocks.
5.  **Saves an annotated image** showing bounding boxes only around the filtered, non-text components.

### `example2.py`: Icon and Button Detection

This script builds upon `example1.py` to further refine the component detection, aiming to isolate small, icon-like elements.
1.  **Performs all steps from `example1.py`** to get a set of non-text components.
2.  **Applies a second filter** to this set based on element size and aspect ratio. This filter is designed to keep elements that are likely to be icons or small buttons.
3.  **Saves an annotated image** showing bounding boxes only around the final filtered components that match the size and shape criteria.

## Component Classifier

The `example.py` script uses a `TestClassifier` class (defined in `classifier.py`) to categorize the detected UI components.

-   **Model**: It loads a pre-trained Keras model (`cnn-generalized.h5`). This model is a convolutional neural network (CNN).
-   **Input**: It takes cropped images of detected components, resizes them to 64x64 pixels, and preprocesses them.
-   **Output**: The model predicts one of the 13 trained classes for each component:
    -   `Button`
    -   `CheckBox`
    -   `Chronometer`
    -   `EditText`
    -   `Image`
    -   `ImageButton`
    -   `NumberPicker`
    -   `RadioButton`
    -   `RatingBar`
    -   `SeekBar`
    -   `Spinner`
    -   `Switch`
    -   `TextView`

As the name `TestClassifier` suggests, this is a proof-of-concept implementation. The accuracy may vary, but it serves as a good starting point for integrating more advanced classification models.
