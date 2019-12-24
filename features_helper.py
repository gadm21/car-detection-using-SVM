
import cv2
import config
from skimage.feature import hog
import numpy as np

'''
feature extraction
'''



def color_space_conversion(image, color_space):

    if color_space != 'RGB':
        if color_space == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image

def bin_spatial(image, size=(32,32)):
    '''
    just returns the image itself, reseized and unrolled in a feature vector
    '''
    features= cv2.resize(image, size).ravel()
    return features

def color_hist(image, nbins= 32, bins_range= (0, 256)):
    '''
    returns the color histogram features of a given image "img"
    Histogram is computed for each channel separately then concatenated
    '''

    channel1_hist= np.histogram(image[:, :, 0], bins= nbins, range= bins_range)
    channel2_hist= np.histogram(image[:, :, 1], bins= nbins, range= bins_range)
    channel3_hist= np.histogram(image[:, :, 2], bins= nbins, range= bins_range)

    hist_features= np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features

def get_hog_features(image, orient, pix_per_cell, cell_per_block, verbose= False, feature_vec= True):
    '''
    return hog features of a given image
    If 'verbose==True', a visualization of the histogram is also returned
    '''
    if verbose:
        features, hog_image= hog(image, orientation= orient, pixels_per_cell= (pix_per_cell, pix_per_cell),
                                cells_per_block= (cell_per_block, cell_per_block),
                                transform_sqrt= True, visualise= verbose,
                                feature_vector= feature_vec)
        return features, hog_image
    else:
        features= hog(image, orientations= orient, pixels_per_cell= (pix_per_cell, pix_per_cell),
                    cells_per_block= (cell_per_block, cell_per_block),
                    transform_sqrt= True, visualise= verbose,
                    feature_vector= feature_vec)
        return features

def image_to_features(image, feature_parameters):
    '''
    extracts and returns the feature vector of an image

    paramters:
    ____________
    image: ndarray
        input image to perform feature extraction on
    feature_paramters: dict
        dictionary of paramters of feature extraction process
    
    returns:
    ____________
    features: ndarray
        array of features of the input image
    '''

    color_space= feature_parameters['color_space']
    spatial_size= feature_parameters['spatial_size']
    hist_bins= feature_parameters['hist_bins']
    orient= feature_parameters['orient']
    pix_per_cell= feature_parameters['pix_per_cell']
    cell_per_block= feature_parameters['cell_per_block']
    hog_channel= feature_parameters['hog_channel']
    spatial_feat= feature_parameters['spatial_feat']
    hist_feat= feature_parameters['hist_feat']
    hog_feat= feature_parameters['hog_feat']

    image_features= []
    
    #apply color space conversion according to paramters
    image= color_space_conversion(image, color_space)

    if spatial_feat:
        spatial_features= bin_spatial(image, size= spatial_size)
        image_features.append(spatial_features)
    
    if hist_feat:
        hist_features= color_hist(image, nbins= hist_bins)
        image_features.append(hist_features)
    
    #I don't understand how is hog_channel compared to "ALL" (string) in the following
    #if condition and used as the channel number (int) in line 98
    if hog_feat:
        if hog_channel == "ALL":
            hog_features= []
            for channel in range(image.shape[2]):
                this_channel_hog_feature= get_hog_features(image[:, :, channel],
                                                            orient, pix_per_cell,
                                                            cell_per_block)
                hog_features.append(this_channel_hog_feature)
            hog_features= np.ravel(hog_features)
        else:
            hog_features= get_hog_features(image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block)
    image_features.append(hog_features)

    return np.concatenate(image_features)

def extract_features_from_file_list(files, feature_parameters):
    '''
    extract features from a list of images

    parameters:
    ___________
    files: list
        list of file paths on which feature extraction process is performed
    feature_parameters: dict
        dictionary of paramters that control feature extraction process

    returns:
    _________________
    features: list
        list of feauture array, one for each image (file)
    '''

    features= []

    for file in files:
        resize_h, resize_w= feature_parameters['resize_h'], feature_parameters['resize_w']
        image= cv2.resize(cv2.imread(file), (resize_h, resize_w))

        image_features= image_to_features(image, feature_parameters)
        features.append(image_features)
    
    return features



'''
feature detection
'''

def set_region_boundaries(x_ss, y_ss, image):
    if x_ss[0] is None: x_ss[0]= 0
    if y_ss[0] is None: y_ss[0]= 0
    if x_ss[1] is None: x_ss[1]= image.shape[1]
    if y_ss[1] is None: y_ss[1]= image.shape[0]

    return x_ss, y_ss 

def create_slide_windows(image, x_start_stop= [None, None], y_start_stop= [None, None],
                xy_window= (64, 64), xy_overlap= (0.5, 0.5)):
    '''
    implementation of a sliding window in the region of interest of an image
    '''

    #if x and/or y start/stop are not defined, set to image boundaries
    x_start_stop, y_start_stop= set_region_boundaries(x_start_stop, y_start_stop, image)

    #determine the span of the region to be searched
    x_span= x_start_stop[1]- x_start_stop[0]
    y_span= y_start_stop[1]- y_start_stop[0]

    #determine the size of step (in pexels)
    x_pix_per_step= np.int(xy_window[0]* (1- xy_overlap[0]))
    y_pix_per_step= np.int(xy_window[1]* (1- xy_overlap[1]))

    #determine the number of windows in x/y
    x_windows= np.int(x_span/ x_pix_per_step) -1
    y_windows= np.int(y_span/ y_pix_per_step) -1

    window_list= []
    for i in range(y_windows):
        for j in range(x_windows):

            #calculate window position
            x_start= j* x_pix_per_step + x_start_stop[0]
            x_end= x_start+ xy_window[0]
            y_start= i* y_pix_per_step + y_start_stop[1]
            y_end= y_start+ xy_window[1]

            window_list.append((x_start, y_start), (x_end, y_end))
    
    return window_list

def search_windows(image, windows, classifier, scaler, paramters):

    #list of positive windows
    hot_windows= []

    resize_h, resize_w= paramters['resize_h'], paramters['resize_w']

    for window in windows:

        #extract the window image from the image
        test_image= cv2.resize(image[window[0][1]: window[1][1],
                                window[0][0]: window[1][0]],
                                (resize_h, resize_w))
        
        test_image_features= image_to_features(test_image, paramters)

        test_image_features_scaled= scaler.transform(np.array(test_image_features).reshape(1, -1))

        prediction= classifier.predict(test_image_features_scaled)
        
        if prediction == 1: hot_windows.append(window)

    return hot_windows

def draw_boxes(image, windows, color= (0, 0, 255), thick=5):
    for window in windows:
        topleft_corner= tuple(window[0])
        bottomright_corner= tuple(window[1])
        cv2.rectangle(image, topleft_corner, bottomright_corner, color, thick)
    
    return image 