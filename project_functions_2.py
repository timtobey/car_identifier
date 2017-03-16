import os
import glob
import numpy as np
from skimage.feature import hog
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2


def get_data(image_root_path):  # get image path names for HOG
    search_path = os.path.join(image_root_path, "test_data", )
    images = glob.glob(search_path + "/*/*/*.png")
    cars = []
    notcars = []
    for image in images:
        if 'not_car' in image:
            notcars.append(image)
        else:
            cars.append(image)
    np.save("car_image_file_names", np.asarray(cars))
    np.save("not_car_image_file_names", np.asarray(notcars))


# Get Histogram of Oriented Gradients (HOG)
def hog_my_image(image, orientations, pix_per_cell,
                 cell_per_block, visualise, feature_vector):
    features = hog(image, orientations, pixels_per_cell=(pix_per_cell, pix_per_cell),
                   cells_per_block=(cell_per_block, cell_per_block),
                   transform_sqrt=False, visualise=visualise, feature_vector=feature_vector)
    return features


# Convert images to desired color space
def convert_image_to_desired_color_space(image, color_space):
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(image)
    return feature_image

# Spatial Binning on an image
def bin_spatial(feature_image, spatial_size):
    color1 = cv2.resize(feature_image[:, :, 0], spatial_size).ravel()
    color2 = cv2.resize(feature_image[:, :, 1], spatial_size).ravel()
    color3 = cv2.resize(feature_image[:, :, 2], spatial_size).ravel()
    features = np.hstack((color1, color2, color3))
    return features


# Create Color Histogram of Color
def color_hist(feature_image, hist_bins, bins_range):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(feature_image[:, :, 0], bins=hist_bins, range=bins_range)
    channel2_hist = np.histogram(feature_image[:, :, 1], bins=hist_bins, range=bins_range)
    channel3_hist = np.histogram(feature_image[:, :, 2], bins=hist_bins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    color_hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return color_hist_features


# Choose what type of HOG function to use: Visualize True or False
def hog_image(feature_image, hog_channel, pix_per_cell, orientations, cell_per_block, feature_vec):
    if hog_channel == 'ALL':
        hog_features = []
        for channel in range(feature_image.shape[2]):
            hog_features.append(hog_my_image(feature_image[:, :, channel],
                                             orientations, pix_per_cell, cell_per_block,
                                             visualise=False, feature_vector=feature_vec, ))
        hog_features = np.ravel(hog_features)
    else:
        hog_features = hog_my_image(feature_image[:, :, hog_channel], orientations,
                                    pix_per_cell, cell_per_block, visualise=False,
                                    feature_vector=feature_vec)
    # Append the new feature vector to the features list
    return hog_features
    # file_features.append(hog_features)


# process a single image through the image pipe for vector machine
def single_image__feature_pipe(image, color_space, spatial_size, hist_bins, bins_range, hog_channel,
                               pix_per_cell, orientations, cell_per_block, feature_vec):
    image = mpimg.imread(image)  # read image
    single_image_feature_histogram = []  # Create empty list
    feature_image = convert_image_to_desired_color_space(image, color_space)  # Convert to desired color
    spatial_features = bin_spatial(feature_image, spatial_size)  # spatial bin the image
    single_image_feature_histogram.append(spatial_features)  # append data to feature list
    color_hist_features = color_hist(feature_image, hist_bins, bins_range)  # color histogram
    single_image_feature_histogram.append(color_hist_features)  # append to feature list
    # hog image
    hog_features = hog_image(feature_image, hog_channel, pix_per_cell, orientations, cell_per_block, feature_vec)
    single_image_feature_histogram.append(hog_features)  # add features to list
    return single_image_feature_histogram


# Process list of data images and return image feature
def all_image_process(image_list, color_space, spatial_size, hist_bins, bins_range, hog_channel,
                      pix_per_cell, orientations, cell_per_block, feature_vec):
    all_image_features = []
    for image in image_list:
        single_image_feature_histogram = single_image__feature_pipe(image, color_space, spatial_size,
                                                                    hist_bins, bins_range, hog_channel, pix_per_cell,
                                                                    orientations, cell_per_block, feature_vec)
        all_image_features.append(np.concatenate(single_image_feature_histogram))

    return all_image_features


# process image to find cars and return box
def find_cars(image, color_space, ystart, ystop, svc, orientations,
              pix_per_cell, cell_per_block, spatial_size, hist_bins, bins_range, X_scaler):
    box_list = []  # array for bound boxes to be used with heat map
    img = image.astype(np.float32) / 255

    for scale in range(100, 175, 25):  # range to scale image
        # print(scale)
        scale = scale / 100  # convert scale to decimal for calculation

        img_tosearch = img[ystart:ystop, :, :]  # Area to Search
        ctrans_tosearch = convert_image_to_desired_color_space(img_tosearch,
                                                               color_space)  # convert_color(img_tosearch, conv='RGB2YCrCb')

        if scale != 1:  # Scale image
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]  # size of one hog channel

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1

        # nfeat_per_block = orientations * cell_per_block ** 2
        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64  # 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog_channel = 0
        hog1 = hog_image(ctrans_tosearch, hog_channel, pix_per_cell, orientations, cell_per_block, feature_vec=False)
        hog_channel = 1
        hog2 = hog_image(ctrans_tosearch, hog_channel, pix_per_cell, orientations, cell_per_block, feature_vec=False)
        hog_channel = 2
        hog3 = hog_image(ctrans_tosearch, hog_channel, pix_per_cell, orientations, cell_per_block, feature_vec=False)
        # window over the image

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

                # Get color features
                spatial_features = bin_spatial(subimg, spatial_size)  # bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, hist_bins, bins_range)  # color_hist(subimg, nbins=hist_bins)
                # stack data and scale
                test_features = X_scaler.transform(
                    np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
                # tesdt features
                test_prediction = svc.predict(test_features)
                # if precition is true create box around item
                if test_prediction == 1:
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    pt1 = (xbox_left, ytop_draw + ystart)
                    pt2 = (xbox_left + win_draw, ytop_draw + win_draw + ystart)
                    box_list = bound_boxes(box_list, pt1, pt2)

    return box_list


# Create a list of indtification boxes
def bound_boxes(box_list, pt1, pt2):
    box_list.append((pt1, pt2))
    return box_list


# add "heat" to each box found
def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


# zero out pixels that are below the heat threshold
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return a thresholded map
    mpimg.imsave("heatmap.jpg",heatmap)
    return heatmap


# draw boxes on image
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# plot examples
def plot_examples(image, converted_image):
    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(image)
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(converted_image)
    plt.title('HOG Visualization')
    plt.show(block=True)
