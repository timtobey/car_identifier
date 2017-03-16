from project_functions import *
from sklearn.preprocessing import StandardScaler
import time
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import pickle

# get car and not car image file names
# save image names to list
# Comment out so not to run each time
# image_root_path = "G:/"
# get_data(image_root_path)

#############################################################################
# Settings for:
# Spatial Binning on an image
# Color Histogram
# Get Histogram of Oriented Gradients (HOG)
############################################################################

orientations = 9
pix_per_cell = 8
cell_per_block = 2
spatial_size = (32, 32)
hist_bins = 32
bins_range = (0.0, 1)  # As impages are png, bins range is set to this
visualise = False
feature_vec = True
transform_sqrt = False
hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
color_space = 'YCrCb'

############################################################
# Process Images
# 1 Spatial Binning
# 2 Color Histogram
# 3 Histogram of Oriented Gradients (HOG)
#  Return list of Features
###########################################################

# Process Car images
image_list = np.load("car_image_file_names.npy")
car_features = all_image_process(image_list, color_space, spatial_size, hist_bins,
                                 bins_range, hog_channel, pix_per_cell,
                                 orientations, cell_per_block, feature_vec)

# Process Non-Car Images
image_list = np.load("not_car_image_file_names.npy")
notcar_features = all_image_process(image_list, color_space, spatial_size, hist_bins,
                                    bins_range, hog_channel, pix_per_cell,
                                    orientations, cell_per_block, feature_vec)

# stack car and non-car features together
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

print('Using:', orientations, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
svc = LinearSVC()
# Check the training time for the SVC
t = time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2 - t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
# Check the prediction time for a single sample
t = time.time()

###################################################################
# Create Pickle of SVM, Scaler and Setting used
# This is done so images can be processed without training the data
#      each time.
####################################################################

# Model Settings
pickle_dictionary = {}
pickle_dictionary["svc"] = svc
pickle_dictionary["X_scaler"] = X_scaler
pickle_dictionary["orientations"] = orientations
pickle_dictionary["pix_per_cell"] = pix_per_cell
pickle_dictionary["cell_per_block"] = cell_per_block
pickle_dictionary["hist_bins"] = hist_bins
pickle_dictionary["spatial_size"] = spatial_size
pickle_dictionary["visualise"] = visualise
pickle_dictionary["feature_vec"] = feature_vec
pickle_dictionary["transform_sqrt"] = transform_sqrt
pickle_dictionary["hog_channel"] = hog_channel
pickle_dictionary["color_space"] = color_space
pickle.dump(pickle_dictionary, open("trained_svc.p", "wb"))  # save pickle
