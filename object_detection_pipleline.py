import cv2
import numpy as np
from config import features_extraction_parameters
from skimage.feature import hog
from features_helper import color_space_conversion, get_hog_features, bin_spatial, color_hist
import time
import scipy
import collections
import matplotlib.pyplot as plt
from general_helper import load

time_window = 5
hot_windows_history = collections.deque(maxlen=time_window)


def draw_boxes(img, bbox_list, color=(0, 0, 255), thick=6):
    """
    Draw all bounding boxes in `bbox_list` onto a given image.
    :param img: input image
    :param bbox_list: list of bounding boxes
    :param color: color used for drawing boxes
    :param thick: thickness of the box line
    :return: a new image with the bounding boxes drawn
    """
    # Make a copy of the image
    img_copy = np.copy(img)

    # Iterate through the bounding boxes
    for bbox in bbox_list:
        # Draw a rectangle given bbox coordinates
        tl_corner = tuple(bbox[0])
        br_corner = tuple(bbox[1])
        cv2.rectangle(img_copy, tl_corner, br_corner, color, thick)

    # Return the image copy with boxes drawn
    return img_copy
    

def draw_labeled_bounding_boxes(img, labeled_frame, num_objects):
    """
    Starting from labeled regions, draw enclosing rectangles in the original color frame.
    """
    # Iterate through all detected cars
    for car_number in range(1, num_objects+1):

        # Find pixels with each car_number label value
        rows, cols = np.where(labeled_frame == car_number)

        # Find minimum enclosing rectangle
        x_min, y_min = np.min(cols), np.min(rows)
        x_max, y_max = np.max(cols), np.max(rows)

        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=(255, 0, 0), thickness=6)

    return img
    
    
def normalize_image(img):
    """
    Normalize image between 0 and 255 and cast to uint8
    (useful for visualization)
    """
    img = np.float32(img)

    img = img / img.max() * 255

    return np.uint8(img)

def compute_heatmap_from_detections(frame, hot_windows, threshold=5, verbose=False):
    """
    Compute heatmaps from windows classified as positive, in order to filter false positives.
    """
    h, w, c = frame.shape

    heatmap = np.zeros(shape=(h, w), dtype=np.uint8)

    for bbox in hot_windows:
        # for each bounding box, add heat to the corresponding rectangle in the image
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        heatmap[y_min:y_max, x_min:x_max] += 1  # add heat

    # apply threshold + morphological closure to remove noise
    _, heatmap_thresh = cv2.threshold(heatmap, threshold, 255, type=cv2.THRESH_BINARY)
    heatmap_thresh = cv2.morphologyEx(heatmap_thresh, op=cv2.MORPH_CLOSE,
                                      kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13)), iterations=1)
    if verbose:
        f, ax = plt.subplots(1, 3)
        ax[0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ax[1].imshow(heatmap, cmap='hot')
        ax[2].imshow(heatmap_thresh, cmap='hot')
        plt.show()

    return heatmap, heatmap_thresh
    
    
def find_cars(image, y_start, y_stop, scale, svc, feature_scaler, feat_extr_params):
    """
    Extract features from the input image using hog sub-sampling and make predictions on these.

    Parameters
    ----------
    image : ndarray
        Input image.
    y_start : int
        Lower bound of detection area on 'y' axis.
    y_stop : int
        Upper bound of detection area on 'y' axis.
    scale : float
        Factor used to subsample the image before feature extraction.
    svc : Classifier
        Pretrained classifier used to perform prediction of extracted features.
    feature_scaler : sklearn.preprocessing.StandardScaler
        StandardScaler used to perform feature scaling at training time.
    feat_extr_params : dict
        dictionary of parameters that control the process of feature extraction.

    Returns
    -------
    hot_windows : list
        list of bounding boxes (defined by top-left and bottom-right corners) in which cars have been detected
    """
    hot_windows = []

    resize_h = feat_extr_params['resize_h']
    resize_w = feat_extr_params['resize_w']
    color_space = feat_extr_params['color_space']
    spatial_size = feat_extr_params['spatial_size']
    hist_bins = feat_extr_params['hist_bins']
    orient = feat_extr_params['orient']
    pix_per_cell = feat_extr_params['pix_per_cell']
    cell_per_block = feat_extr_params['cell_per_block']

    draw_img = np.copy(image)

    image_crop = image[y_start:y_stop, :, :]
    image_crop = color_space_conversion(image_crop, color_space=color_space)

    if scale != 1:
        imshape = image_crop.shape
        image_crop = cv2.resize(image_crop, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = image_crop[:, :, 0]
    ch2 = image_crop[:, :, 1]
    ch3 = image_crop[:, :, 2]

    # Define blocks and steps as above
    n_x_blocks = (ch1.shape[1] // pix_per_cell) - 1
    n_y_blocks = (ch1.shape[0] // pix_per_cell) - 1

    # 64 was the original sampling rate, with 8 cells and 8 pix per cell
    window = 64
    n_blocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 4  # Instead of overlap, define how many cells to step
    n_x_steps = (n_x_blocks - n_blocks_per_window) // cells_per_step
    n_y_steps = (n_y_blocks - n_blocks_per_window) // cells_per_step

    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(n_x_steps):
        for yb in range(n_y_steps):
            y_pos = yb * cells_per_step
            x_pos = xb * cells_per_step

            # Extract HOG for this patch
            hog_feat1 = hog1[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_feat2 = hog2[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_feat3 = hog3[y_pos:y_pos + n_blocks_per_window, x_pos:x_pos + n_blocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            x_left = x_pos * pix_per_cell
            y_top = y_pos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(image_crop[y_top:y_top + window, x_left:x_left + window], (resize_w, resize_h))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = feature_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(x_left * scale)
                ytop_draw = np.int(y_top * scale)
                win_draw = np.int(window * scale)
                tl_corner_draw = (xbox_left, ytop_draw + y_start)
                br_corner_draw = (xbox_left + win_draw, ytop_draw + win_draw + y_start)

                cv2.rectangle(draw_img, tl_corner_draw, br_corner_draw, (0, 0, 255), 6)

                hot_windows.append((tl_corner_draw, br_corner_draw))

    return hot_windows
    
def prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection):

    h, w, c = frame.shape

    # decide the size of thumbnail images
    thumb_ratio = 0.25
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    # resize to thumbnails images from various stages of the pipeline
    thumb_hot_windows = cv2.resize(img_hot_windows, dsize=(thumb_w, thumb_h))
    thumb_heatmap = cv2.resize(img_heatmap, dsize=(thumb_w, thumb_h))
    thumb_labeling = cv2.resize(img_labeling, dsize=(thumb_w, thumb_h))

    off_x, off_y = 20, 45

    # add a semi-transparent rectangle to highlight thumbnails on the left
    mask = cv2.rectangle(img_detection.copy(), (0, 0), (2*off_x + thumb_w, h), (0, 0, 0), thickness=cv2.FILLED)
    img_blend = cv2.addWeighted(src1=mask, alpha=0.2, src2=img_detection, beta=0.8, gamma=0)

    # stitch thumbnails
    img_blend[off_y:off_y+thumb_h, off_x:off_x+thumb_w, :] = thumb_hot_windows
    img_blend[2*off_y+thumb_h:2*(off_y+thumb_h), off_x:off_x+thumb_w, :] = thumb_heatmap
    img_blend[3*off_y+2*thumb_h:3*(off_y+thumb_h), off_x:off_x+thumb_w, :] = thumb_labeling

    return img_blend


def process_pipeline(frame, svc, feature_scaler, feat_extraction_params, keep_state=True, verbose=False):

    hot_windows = []

    for subsample in np.arange(1, 3, 0.5):
        hot_windows += find_cars(frame, 400, 600, subsample, svc, feature_scaler, feat_extraction_params)

    if keep_state:
        if hot_windows:
            hot_windows_history.append(hot_windows)
            hot_windows = np.concatenate(hot_windows_history)

    # compute heatmaps positive windows found
    thresh = (time_window - 1) if keep_state else 0
    heatmap, heatmap_thresh = compute_heatmap_from_detections(frame, hot_windows, threshold=thresh, verbose=False)
                                                 
    # label connected components
    labeled_frame, num_objects = scipy.ndimage.measurements.label(heatmap_thresh)

    # prepare images for blend
    img_hot_windows = draw_boxes(frame, hot_windows, color=(0, 0, 255), thick=2)                 # show pos windows
    img_heatmap = cv2.applyColorMap(normalize_image(heatmap), colormap=cv2.COLORMAP_HOT)         # draw heatmap
    img_labeling = cv2.applyColorMap(normalize_image(labeled_frame), colormap=cv2.COLORMAP_HOT)  # draw label
    img_detection = draw_labeled_bounding_boxes(frame.copy(), labeled_frame, num_objects)        # draw detected bboxes

    img_blend_out = prepare_output_blend(frame, img_hot_windows, img_heatmap, img_labeling, img_detection)

    if verbose:
        cv2.imshow('detection bboxes', img_hot_windows)
        cv2.imshow('heatmap', img_heatmap)
        cv2.imshow('labeled frame', img_labeling)
        cv2.imshow('detections', img_detection)
        cv2.waitKey()

    return img_blend_out
    
if __name__ == "__main__":
    
    svc= load("classifier")
    feature_scaler= load("feature_scaler")
    frame= cv2.imread("test4.jpg")
    out_image= process_pipeline(frame, svc, feature_scaler, features_extraction_parameters, keep_state=False, verbose=False)
    cv2.imshow("h", out_image)
    cv2.waitKey(0)
    cv2.destroyWindow("h")