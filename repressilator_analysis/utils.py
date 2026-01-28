import numpy as np
def map_mask_to_image(mask, image, value):
    rows=mask[:,0]
    cols=mask[:,1]
    copy_img=image.copy()
    copy_img[rows, cols]=value
    return copy_img
def get_centroid(mask):
    centroid_row = np.mean(mask[:, 0])                                                                                                                                                                
    centroid_col = np.mean(mask[:, 1])
    return [centroid_row, centroid_col]
def check_position_dupes(centroid, centroid_list, threshold=3):
    distances=[np.linalg.norm(centroid-x) for x in centroid_list]
    return any([x<threshold for x in distances])
def RMSE(x,y):
    return np.sqrt(np.mean(np.square(np.subtract(x,y))))