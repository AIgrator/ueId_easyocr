from os.path import join as pjoin
import os


class Config:

    def __init__(self):
        # Model paths
        self.CNN_PATH = 'models/cnn-rico-1.h5'
        self.EAST_PATH = 'models/east_icdar2015_resnet_v1_50_rbox'
        self.MODEL_PATH_TEXT = 'models/cnn-textview-2.h5'
        self.MODEL_PATH_NOISE = 'models/cnn-noise-1.h5'
        self.MODEL_PATH_IMAGE = 'models/cnn-image-1.h5'

        # Class maps
        self.element_class = ['Button', 'CheckBox', 'Chronometer', 'EditText', 'ImageButton', 'ImageView',
                              'ProgressBar', 'RadioButton', 'RatingBar', 'SeekBar', 'Spinner', 'Switch',
                              'ToggleButton', 'VideoView', 'TextView']
        self.class_number = len(self.element_class)
        self.CLASS_MAP_TEXT = ['Text', 'Non-Text']
        self.CLASS_MAP_NOISE = ['Noise', 'Non-Noise']
        self.CLASS_MAP_IMAGE = ['Image', 'Non-Image']

        self.COLOR = {'Button': (0, 255, 0), 'CheckBox': (0, 0, 255), 'Chronometer': (255, 166, 166),
                      'EditText': (255, 166, 0),
                      'ImageButton': (77, 77, 255), 'ImageView': (255, 0, 166), 'ProgressBar': (166, 0, 255),
                      'RadioButton': (166, 166, 166),
                      'RatingBar': (0, 166, 255), 'SeekBar': (0, 166, 10), 'Spinner': (50, 21, 255),
                      'Switch': (80, 166, 66), 'ToggleButton': (0, 66, 80), 'VideoView': (88, 66, 0),
                      'TextView': (169, 255, 0), 'NonText': (0,0,255),
                      'Compo':(0, 0, 255), 'Text':(169, 255, 0), 'Block':(80, 166, 66)}

    def build_output_folders(self):
        # setting data flow paths
        self.ROOT_INPUT = "E:\\Mulong\\Datasets\\rico\\combined"
        self.ROOT_OUTPUT = "E:\\Mulong\\Result\\rico\\rico_uied\\rico_new_uied_v3"

        self.ROOT_IMG_ORG = pjoin(self.ROOT_INPUT, "org")
        self.ROOT_IP = pjoin(self.ROOT_OUTPUT, "ip")
        self.ROOT_OCR = pjoin(self.ROOT_OUTPUT, "ocr")
        self.ROOT_MERGE = pjoin(self.ROOT_OUTPUT, "merge")
        self.ROOT_IMG_COMPONENT = pjoin(self.ROOT_OUTPUT, "components")
        if not os.path.exists(self.ROOT_IP):
            os.mkdir(self.ROOT_IP)
        if not os.path.exists(self.ROOT_OCR):
            os.mkdir(self.ROOT_OCR)
        if not os.path.exists(self.ROOT_MERGE):
            os.mkdir(self.ROOT_MERGE)
