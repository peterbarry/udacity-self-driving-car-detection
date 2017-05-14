
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


from features import extract_features

# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split



def load_and_train_all_features():
    # Read in car and non-car images

    non_vehicle_images = glob.glob('training_data/non-vehicles/*/*.png')
    vehicle_images = glob.glob('training_data/vehicles/*/*.png')
    train_jpeg=False
    #non_vehicle_images = glob.glob('training_data/non-vehicles_smallset/*/*.jpeg')
    #vehicle_images = glob.glob('training_data/vehicles_smallset/*/*.jpeg')
    #train_jpeg=True
    #print('USING SMALLSET JPEG IMAGES!!!! us mping to load in features')
    
    print ('Trainig Data:')
    print('Car images:', len(vehicle_images))
    print('Non-Car images:', len(non_vehicle_images))


    cars = []
    notcars = []

    for image in vehicle_images:
            cars.append(image)

    for image in non_vehicle_images:
            notcars.append(image)

    # experemted with play with these values to see how your classifier
    # performs under different binning scenarios


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
    train_jpeg=False


    print('Generating Features for Cars')
    car_features = extract_features(cars, color_space=colorspace, spatial_size=(spatial, spatial),
                            hist_bins=hist_bins, hist_range=(0, 256),orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,train_jpeg=train_jpeg)

    print('Generating Features for non car images')

    notcar_features = extract_features(notcars, color_space=colorspace, spatial_size=(spatial, spatial),
                            hist_bins=hist_bins, hist_range=(0, 256),orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, hog_channel=hog_channel,
                            spatial_feat=spatial_feat, hist_feat=hist_feat, hog_feat=hog_feat,train_jpeg=train_jpeg)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using spatial binning of:',spatial,
        'and', hist_bins,'histogram bins')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC(C=1.0)
    #clf = CalibratedClassifierCV(svc) # performance was same on small sample set.
    clf = svc
    # Check the training time for the SVC
    t=time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts:     ', clf.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

    print('saving model to classifier-svm.pkl')
    joblib.dump(clf, 'models/classifer-svm.pkl')
    joblib.dump(X_scaler, 'models/xscaler.pkl')


    return clf,X_scaler

clf = load_and_train_all_features()
