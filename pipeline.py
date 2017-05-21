
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip

import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from window import *
from scipy.ndimage.measurements import label



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
draw_raw_clasification_boxes = True
add_inter_frame_processing = False
draw_frame_hotbox=False

#move this to a class.
number_of_frames_history = 4
interframe_list_of_boxes=[]

print('Model loaded')

def process_image(image):

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

            if draw_raw_clasification_boxes == True:
                window_img = draw_boxes(draw_image, hot_windows, color=(0, 255, 255), thick=2)
                draw_image = window_img



            print('mum of hot windows:')
            print(len(hot_windows))
            print('add interframe processing')
            print(add_inter_frame_processing)
            # Generate a collection of windows-inter frame
            if add_inter_frame_processing == True:
                #Adding heatmap summary box
                heat_frame = np.zeros_like(image[:,:,0]).astype(np.float)
                heat_frame = add_heat(heat_frame,hot_windows)
                    # Apply threshold to help remove false positives
                heat_frame = apply_heat_threshold(heat_frame,1)

                    # Visualize the heatmap when displaying
                heatmap = np.clip(heat_frame, 0, 255)
                # Find final boxes from heatmap using label function
                labels = label(heatmap)
                boxes = get_boxes_from_labels(labels)
                print('Windows found in this frame')
                print(len(boxes))


                if draw_frame_hotbox == True:
                        window_img = draw_boxes(draw_image, boxes, color=(0, 255, 0), thick=2)
                        draw_image = window_img

                interframe_list_of_boxes.append(boxes)
                if len(interframe_list_of_boxes) > number_of_frames_history:
                    interframe_list_of_boxes.pop(0)

                #concatante list of windows over multiple frames.
                print('len of interframe array')
                print(len(interframe_list_of_boxes))
                hot_windows_list = []
                for list in interframe_list_of_boxes:
                    for wind in list:
                        hot_windows_list.append(wind)
            else:
                hot_windows_list = hot_windows

            print('mum of interfarame hot windows:')
            print(len(hot_windows_list))


            #Adding heatmap summary box
            heat = np.zeros_like(image[:,:,0]).astype(np.float)
            #window_img = draw_boxes(window_img, windows, color=(0, 255, 0), thick=2)
            # Add heat to each box in box list
            heat = add_heat(heat,hot_windows_list)
                                # Apply threshold to help remove false positives
            heat = apply_heat_threshold(heat,1)

                                # Visualize the heatmap when displaying
            heatmap = np.clip(heat, 0, 255)

            # Find final boxes from heatmap using label function
            labels = label(heatmap)
            window_img = draw_labeled_bboxes(draw_image, labels)

            return window_img


def process_test_images():

        test_jpeg = glob.glob('test_images/*.jpg')
        for img_name in test_jpeg:
            print(img_name)
            image = mpimg.imread(img_name)
            out_image = process_image(image)

            names = img_name.split('.')
            print ('saving:1')
            base = 'output_images/' + names[0]+'-detect.png'
            cv2.imwrite(base,out_image)





def process_videofile():
    print("Running on test video1...")

    #####################################
    # Run our pipeline on the test video
    #####################################
    clip = VideoFileClip("./test_video.mp4")
    output_video = "./test_video_output.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

    clip = VideoFileClip("./project_video.mp4")
    output_video = "./project_video_output.mp4"
    output_clip = clip.fl_image(process_image)
    output_clip.write_videofile(output_video, audio=False)

add_inter_frame_processing = False
draw_raw_clasification_boxes=True
draw_frame_hotbox=False
process_test_images()
add_inter_frame_processing = True
draw_raw_clasification_boxes=False
draw_frame_hotbox=True
process_videofile()
