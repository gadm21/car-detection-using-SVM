import cv2
import numpy as np
from general_helper import  load
from features_helper import image_to_features
from config import features_extraction_parameters

test_image= cv2.imread("test_image2.png")
test_image_features= image_to_features(test_image, features_extraction_parameters)
test_image_features= np.reshape(test_image_features, (1, len(test_image_features)))
new_svc= load("classifier")
cc= new_svc.predict(test_image_features)
print(cc)