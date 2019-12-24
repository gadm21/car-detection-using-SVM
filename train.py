import numpy as np
import os
import cv2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt 

from general_helper import get_file_list_recursively, save, load
#from config import car_dataset, notcar_dataset
from features_helper import extract_features_from_file_list, create_slide_windows, image_to_features
from features_helper import search_windows, draw_boxes
from config import features_extraction_parameters


    
if __name__ == "__main__":

    #read paths of car and notcar images (file names not actual image objects)
    cars= get_file_list_recursively(features_extraction_parameters['cars_root_data'], ['.png'],verbose= False)
    notcars= get_file_list_recursively(features_extraction_parameters['notcars_root_data'],['.png'], verbose= False)
    print("cars:", cars[0])
    cars_features= extract_features_from_file_list(cars,features_extraction_parameters)
    notcars_features = extract_features_from_file_list(notcars,features_extraction_parameters)

    #stack cars & notcars data vertically in one matrix
    X= np.vstack((cars_features, notcars_features)).astype(np.float64)
    

    '''
    standarize features
        StandardScaler().fit(X) calculates the mean and std for data in X (here, per column)
        then feature_scaler.transform(X) applies the calculated mean and std in fit(X) to the 
        data in transform(X). Note that the data don't have to be identical
        in both functions. The first data used to calculate the mean and std 
        (usually huge quantity of data is used)
        and the second are the actual data we're working on so we apply standardization on them.
    '''
    feature_scaler= StandardScaler().fit(X)
    scaled_X= feature_scaler.transform(X)

    #create labels vector
    y= np.hstack((np.ones(len(cars_features)), np.zeros(len(notcars_features))))

    #split data into training and testing sets
    x_train, x_test, y_train, y_test= train_test_split(scaled_X, y, test_size=0.2,
                                         random_state= np.random.randint(0, 100))
    
    #define a linear support vector machine classifier
    svc= LinearSVC()
    
    #train the classifier
    svc.fit(x_train, y_train)

    #check accuracy
    print(svc.score(x_test, y_test), 4)

    #save !
    # we're saving feature_scalar to use it's saved paramters (mean&std) 
    # with other datasets of the same categories (cars & notcars)
    save(svc, "classifier")
    save(feature_scaler, "feature_scaler") 
    save(features_extraction_parameters, "features_extraction_paramters")

op()