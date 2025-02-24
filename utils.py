import json
import cv2
import glob
import os
import numpy as np


def explore_dataset(dataset_folder: str,
                    verbose: bool = False,
                    visualize: bool = False) -> None:
    """ Check dataset folder to see sample label, polygon coordinates, bounding boxes, annotation type, roi, etc.
    Args:
        dataset_folder (str): Path of directory that contains images and annotation files.
        verbose (bool): Whether to print out information for each sample
                        Defaults to False
        visualize (bool): Whether to check images and corresponding annotations with visualization.
                          Defaults to False
    Returns:
        None
    """
    image_paths = glob.glob(os.path.join(dataset_folder, '*.bmp'))
    json_paths  = glob.glob(os.path.join(dataset_folder, '*.json'))
    assert len(image_paths) == len(json_paths), 'No file matching 1 by 1'
    for image_path, json_path in zip(image_paths, json_paths):

        # Read json file and extract features
        with open(json_path, 'r', encoding='utf-8') as file:
            annotation = json.load(file)
        label               = annotation['shapes'][0]['label']
        polygon_coordinates = annotation['shapes'][0]['points']
        bounding_box        = annotation['shapes'][0]['bbox']
        annotation_type     = annotation['shapes'][0]['shape_type']
        roi = annotation['rois'][0]

        # Read image
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Draw polygon annotation
        polygon_coordinates_np = np.array(polygon_coordinates, np.int32).reshape((-1, 1, 2)) # (width, height)
        cv2.polylines(image, [polygon_coordinates_np], isClosed=True, color=(0, 255, 0), thickness=2)

        x_poly, y_poly, width_poly, height_poly = cv2.boundingRect(polygon_coordinates_np)
        x_min_poly = x_poly
        y_min_poly = y_poly
        x_max_poly = x_poly + width_poly
        y_max_poly = y_poly + height_poly

        cv2.rectangle(image, (x_min_poly, y_min_poly), (x_max_poly, y_max_poly), color=(255, 0, 0), thickness=2)
        cv2.putText(image, f"{label}", (x_poly, y_poly - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw ROI : (x_min, y_min, x_max, y_max)
        roi_x_min = roi[0]
        roi_y_min = roi[1]
        roi_x_max = roi[2]
        roi_y_max = roi[3]
        cv2.rectangle(image, (roi_x_min, roi_y_min), (roi_x_max, roi_y_max), color=(255, 255, 255), thickness=2)
        cv2.putText(image, "ROI", (roi_x_min, roi_y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 0, 255), 2, cv2.LINE_AA)

        if verbose:
            print(f'Sample label: {label}')

        if visualize:
            cv2.imshow(f'{os.path.basename(image_path)}', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    print('Dataset exploration finished.')
    return None
