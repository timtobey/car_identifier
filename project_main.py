from project_functions_2 import *
import pickle
from moviepy.editor import *
from IPython.display import HTML
from scipy.ndimage.measurements import label

# load Saved SVM data
pickle_dictionary = pickle.load( open( "trained_svc.p", "rb" ) )
svc =pickle_dictionary ["svc"]
X_scaler = pickle_dictionary  ["X_scaler"]
orientations = pickle_dictionary ["orientations"]
pix_per_cell =pickle_dictionary ["pix_per_cell"]
cell_per_block =pickle_dictionary ["cell_per_block"]
hist_bins = pickle_dictionary ["hist_bins"]
spatial_size =pickle_dictionary ["spatial_size"]
visualise = pickle_dictionary ["visualise"]
feature_vec =pickle_dictionary ["feature_vec"]
transform_sqrt =pickle_dictionary ["transform_sqrt"]
hog_channel =pickle_dictionary ["hog_channel"]
color_space = pickle_dictionary["color_space"]



ystart = 400
ystop = 656
scale = 1.5
bins_range = (0,255)
avg_boxes = []
image = mpimg.imread('bbox-example-image.jpg')


def video_pipe_line(image, color_space =color_space, ystart=ystart, ystop = ystop,  svc=svc, orientations =orientations,
              pix_per_cell = pix_per_cell, cell_per_block = cell_per_block, spatial_size =spatial_size, hist_bins =hist_bins,
                    bins_range = bins_range ,X_scaler = X_scaler):

    box_list = find_cars(image, color_space, ystart, ystop,  svc, orientations,
                                  pix_per_cell, cell_per_block, spatial_size, hist_bins, bins_range, X_scaler)
    #creates aggerate list of boxes frames
    avg_boxes.append(box_list)
    if len(avg_boxes) > 12: # drops oldes list of boxes if avg_boxes has more than 12 frames
        avg_boxes.pop(0)
    flatten = sum(avg_boxes, []) #flattens array for heat map

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
        # Add heat to each box in box list
    heat = add_heat(heat, flatten)
# Apply threshold to help remove false positives
    heat = apply_threshold(heat, 2)
    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
# Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img


def process_video(video_path):
    image_name = 'project_video_with_lines.mp4'
    full_image = video_path+image_name
    image = full_image
    yellow_output = video_path+'project_video_with_lines_and_boxes.mp4'
    clip2 = VideoFileClip(image)
    yellow_clip = clip2.fl_image(video_pipe_line)
    yellow_clip.write_videofile(yellow_output, audio=False)
    HTML("""
    <video width="960" height="540" controls>
      <source src="{0}">
    </video>
    """.format(yellow_output))
video_path = "C:/Users/timto/Desktop/video_test/"

process_video(video_path)



