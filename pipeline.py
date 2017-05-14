
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from window import *


print('Loading model ')
clf = joblib.load( 'models/classifer-svm.pkl')
X_scaler=joblib.load('models/xscaler.pkl')


colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial=32
spatial_size = (spatial, spatial) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 656] # Min and max in y to search in slide_window()


print('Model loaded')
def process_test_images():

        test_jpeg = glob.glob('test_images/*.jpg')
        for img_name in test_jpeg:
            print(img_name)
            image = mpimg.imread(img_name)
            draw_image = np.copy(image)

            # Uncomment the following line if you extracted training
            # data from .png images (scaled 0 to 1 by mpimg) and the
            # image you are searching is a .jpg (scaled 0 to 255)
            #image = image.astype(np.float32)/255

            #windows = slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
            #                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

            windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start_stop[0], y_start_stop[1]],
                        xy_window=(256, 256), xy_overlap=(0.5, 0.5))

            windows = windows + slide_window(image, x_start_stop=[None, None], y_start_stop=[y_start_stop[0], 600],
                        xy_window=(128, 128), xy_overlap=(0.8, 0.8))


            windows = windows + slide_window(image, x_start_stop=[300, 1000], y_start_stop=[y_start_stop[0], 480],
                        xy_window=(64, 64), xy_overlap=(0.9, 0.9))


            hot_windows = search_windows(image, windows, clf, X_scaler, color_space=colorspace,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)

            window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
            search_img = draw_boxes(draw_image, windows, color=(0, 0, 255), thick=6)

            names = img_name.split('.')
            print ('saving:1')
            base = 'output_images/' + names[0]+'-detect.png'
            cv2.imwrite(base,window_img)

            base = 'output_images/' + names[0]+'-search.png'
            cv2.imwrite(base,search_img)

process_test_images()
