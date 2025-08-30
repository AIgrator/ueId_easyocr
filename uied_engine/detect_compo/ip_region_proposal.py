import os
import cv2
from os.path import join as pjoin
import time
import json
import numpy as np

from uied_engine.detect_compo.lib_ip import ip_preprocessing as pre
from uied_engine.detect_compo.lib_ip import ip_draw as draw
from uied_engine.detect_compo.lib_ip import ip_detection as det
from uied_engine.detect_compo.lib_ip import file_utils as file
from uied_engine.detect_compo.lib_ip import Component as Compo
from uied_engine.config.CONFIG_UIED import Config
C = Config()


def nesting_inspection(org, grey, compos, ffl_block):
    '''
    Inspect all big compos through block division by flood-fill
    :param ffl_block: gradient threshold for flood-fill
    :return: nesting compos
    '''
    nesting_compos = []
    for i, compo in enumerate(compos):
        if compo.height > 50:
            replace = False
            clip_grey = compo.compo_clipping(grey)
            n_compos = det.nested_components_detection(clip_grey, org, grad_thresh=ffl_block, show=False)
            Compo.cvt_compos_relative_pos(n_compos, compo.bbox.col_min, compo.bbox.row_min)

            for n_compo in n_compos:
                if n_compo.redundant:
                    compos[i] = n_compo
                    replace = True
                    break
            if not replace:
                nesting_compos += n_compos
    return nesting_compos

def detect_components(input_img_path, uied_params, resize_by_height=800, show=False, wai_key=0):
    """
    Detects all components in the image without classifying them.
    Returns the original image, grey image, and the detected components.
    """
    start = time.perf_counter()

    # Step 1: Pre-processing
    org, grey = pre.read_img(input_img_path, resize_by_height)
    binary = pre.binarization(org, grad_min=int(uied_params['min-grad']))

    # Step 2: Element detection
    det.rm_line(binary, show=show, wait_key=wai_key)
    uicompos = det.component_detection(binary, min_obj_area=int(uied_params['min-ele-area']))

    # Step 3: Results refinement
    uicompos = det.compo_filter(uicompos, min_area=int(uied_params['min-ele-area']), img_shape=binary.shape)
    uicompos = det.merge_intersected_compos(uicompos)
    det.compo_block_recognition(binary, uicompos)
    if uied_params['merge-contained-ele']:
        uicompos = det.rm_contained_compos_not_in_block(uicompos)
    Compo.compos_update(uicompos, org.shape)
    Compo.compos_containment(uicompos)

    # Step 4: Nesting inspection
    uicompos += nesting_inspection(org, grey, uicompos, ffl_block=uied_params['ffl-block'])
    Compo.compos_update(uicompos, org.shape)
    
    print("[Component Detection Completed in %.3f s]" % (time.perf_counter() - start))
    return org, grey, uicompos

def classify_components(org_img, components, classifier):
    """
    Classifies a list of components.
    """
    start = time.perf_counter()
    if classifier is not None and 'Elements' in classifier:
        classifier['Elements'].predict([compo.compo_clipping(org_img) for compo in components], components)
    print("[Classification Completed in %.3f s]" % (time.perf_counter() - start))
    return components