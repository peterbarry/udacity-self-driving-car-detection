
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

    #non_vehicle_images = glob.glob('training_data/non-vehicles/*/*.png')
    #vehicle_images = glob.glob('training_data/vehicles/*/*.png')

    non_vehicle_images = glob.glob('training_data/non-vehicles_smallset/*/*.jpeg')
    vehicle_images = glob.glob('training_data/vehicles_smallset/*/*.jpeg')

    print('USING SMALLSET JPEG IMAGES!!!! us mping to load in features')
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
    spatial = 32
    histbin = 32


    print('Generating Features for Cars')
    car_features = extract_features(cars, color_space='RGB', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256),orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True)

    print('Generating Features for non car images')

    notcar_features = extract_features(notcars, color_space='RGB', spatial_size=(spatial, spatial),
                            hist_bins=histbin, hist_range=(0, 256),orient=9,
                            pix_per_cell=8, cell_per_block=2, hog_channel=0,
                            spatial_feat=True, hist_feat=True, hog_feat=True)

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
        'and', histbin,'histogram bins')
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
