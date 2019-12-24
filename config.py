# parameters used in the phase of feature extraction
features_extraction_parameters = {'resize_h': 64,             # resize image height before feat extraction
                          'resize_w': 64,             # resize image height before feat extraction
                          'color_space': 'YCrCb',     # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
                          'orient': 9,                # HOG orientations
                          'pix_per_cell': 8,          # HOG pixels per cell
                          'cell_per_block': 2,        # HOG cells per block
                          'hog_channel': "ALL",       # Can be 0, 1, 2, or "ALL"
                          'spatial_size': (32, 32),   # Spatial binning dimensions
                          'hist_bins': 16,            # Number of histogram bins
                          'spatial_feat': True,       # Spatial features on or off
                          'hist_feat': True,          # Histogram features on or off
                          "cars_root_data" : 'vehicles',
                          "notcars_root_data" : 'non-vehicles',
                          'test_images_dir': "test_images",
                          'hog_feat': True,
                          'mode': "video",
                          'test_video': 'project_video.mp4',
                          'output_video': "output_video.mp4",
                          }           # HOG features on or off
