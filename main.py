
from moviepy.editor import VideoFileClip
from object_detection_pipleline import process_pipeline
import os
import cv2
import matplotlib.pyplot as plt
from config import features_extraction_parameters
from general_helper import load

svc= load("classifier")
feat_extraction_params= load("features_extraction_paramters")
feature_scaler= load("feature_scaler")

def new_process_pipeline(frame):
    
    output_image= process_pipeline(frame, svc, feature_scaler, feat_extraction_params, True, False)
    return output_image

if __name__ == "__main__":


    
    if features_extraction_parameters["mode"] == "video":
        clip= VideoFileClip(features_extraction_parameters["test_video"])
        output_clip= clip.fl_image(new_process_pipeline)
        output_clip.write_videofile(features_extraction_parameters["output_video"], audio= False)
        #plt.plot(features_extraction_parameters["xs"], features_extraction_parameters["ys"])
    else:
        test_images_dir= parameters["test_image_dir"]
        for test_image_file in os.listdir(test_images_dir):
            image= cv2.imread(os.path.join(test_images_dir, test_image_file))
            output_image=  image#lane_finding_pipeline(image)
            
            '''
            cv2.imshow("r", output_image)
            cv2.waitKey(0)
            cv2.destroyWindow("r")
            '''
            
            cv2.imwrite(os.path.join(parameters["output_image_dir"], test_image_file), output_image)
            break
            
            
            
    