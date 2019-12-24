import os
import cv2
import numpy as np
from config import features_extraction_parameters
from features_helper import create_slide_windows, draw_boxes, search_windows
from general_helper import load
import matplotlib.pyplot as plt 


if __name__ == "__main__":

    #test on images
    test_images_dir= features_extraction_parameters['test_images_dir']
    clf= load("classifier")
    feature_scaler= load("feature_scaler")
    test_images_files= os.listdir(test_images_dir)

    for file in test_images_files:
        image= cv2.imread(os.path.join(test_images_dir, file))

        h, w, c= image.shape
        draw_image= np.copy(image)

        windows= create_slide_windows(image, x_start_stop= [None, None], 
         y_start_stop= [h//2, None], xy_window= (64, 64), xy_overlap= (0.8, 0.8))
        
        hot_windows= search_windows(image, windows, clf, feature_scaler, features_extraction_parameters)

        window_image= draw_boxes(draw_image, hot_windows)
    
        plt.imshow(cv2.cvtColor(window_image, cv2.COLOR_BGR2RGB))
        plt.show()