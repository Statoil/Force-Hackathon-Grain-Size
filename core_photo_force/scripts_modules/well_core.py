import os
import lasio
import pandas as pd
import pkg_resources
from collections import defaultdict
from skimage.io import imread, imread_collection, concatenate_images, imshow, imsave
from skimage import feature, color, filters, morphology, segmentation
import matplotlib.pyplot as plt
import numpy as np
from tkinter import *
import random
import pickle


class WellCore(object):
    image_array = None
    start_depth = None
    end_depth = None
    pixels_per_depth_unit = None
    depth_unit = 'M'
    """Create a cored well object from a photo directory, track depth/scale, turk labels, and more"""

    # TODO: Initialize object using a directory of separate photos
    def __init__(self, directory_path, start_depth, end_depth):
        pass
    # TODO: Stitch photo together

    # TODO: Track and save labels

    # TODO: Initialize D3 visualization of core and current labels

    pass

    def _load_image(self, file_path, start_depth, end_depth):
        img = imread(file_path)
        width = img.shape[1]
        height = img.shape[0]
        # check scale

        # set loaded image to merged array at the proper indicies
        


def stitch_images(directory="data", size=128):
    # loop through each core image folder
    plt.interactive(False)
    path = pkg_resources.resource_filename('core_photo_force', directory)
    las_dict = defaultdict(list)
    img_dict = defaultdict(list)
    output = defaultdict(list)
    for subdir, dirs, files in os.walk(path):
        print(files)
        if subdir[-len(directory):] == directory:
            for file in files:
                if file[-3:] == 'las':
                    las_dict[file[:4]] = path + '/' + file
        else:
            for file in files:
                if file[-3:] == 'jpg':

                    img_dict[file[:4]].append(subdir + '/' + file)
    for key, las in las_dict.items():
        las_df = get_grain_size_las(las)
        las_dict[key] = las_df
    for key_well, well in img_dict.items():
        merged_image  = merge_well_images(well) #, depth_to_image_index, image_index_to_depth
        callbreak = False
        for idx in range(100):
            index = random.randrange(75, merged_image.shape[0]-75, 1)
            turk_img = merged_image[index - 75: index + 75, :, :]
            current_plt = plt.imshow(turk_img)
            plt.show()
            notdone = True
            while notdone:
                imshow(turk_img)
                print('enter done here to quit doing this core, enter back at anytime to restart')
                image_good = input('Image is good? 1 is yes 0 is no')
                if image_good == 'back':
                    continue
                if image_good == 'done':
                    callbreak = True
                    break
                if image_good == '1':
                    is_sand = input('Image is sand? 1 is yes 0 is no')
                    if image_good == 'back':
                        continue
                    if is_sand == '1':

                        grain_size = input('Grain size 3-10 :')
                        if grain_size == 'back':
                            continue

                    else:
                        grain_size = 0

                    print('Sed Structure codes : 1 = Massive/None,'
                          ' 2 = Laminated,'
                          ' 3 = Cross Stratification,'
                          ' 4 = burrows')
                    sed_facies = input('Input Sed Structure Code: ')
                else:
                    is_sand = None
                    grain_size = None
                    sed_facies = None
                notdone = False
            if callbreak:
                break
            output['img_Sample'].append(turk_img)
            output['is_good'].append(image_good)
            output['is_sand'].append(is_sand)
            output['grain_size'].append(grain_size)
            output['sed_structure_code'].append(sed_facies)
            plt.close()

        if len(output['img_Sample']) > 0:
            with open('turk_file' + key_well + '.pkl', 'wb') as f:
                pickle.dump(output, f)
            print('here')


def get_grain_size_las(path):
    w6406 = lasio.read(path)
    print(w6406.keys())
    depth = w6406['DEPT']
    grain_size = w6406['GRAIN_SIZE']
    w6406 = pd.DataFrame([depth, grain_size])
    w6406 = w6406.T
    w6406.head(5)
    w6406.columns = ['depth', 'gs']
    w64 = w6406[w6406 > 0].dropna()
    return w64


def merge_well_images(well: list):
    output_img_list = []
    output_img_dict = defaultdict(list)
    cum_height = 0
    for idx, image in enumerate(well):
        print(idx / len(well))
        img = imread(image)
        width = img.shape[1]
        height = img.shape[0]
        mid = int(width / 2)
        output_img = img[:, mid - 75:mid + 75, :]
        output_img_list.append(output_img)
        image_split = image.split('.')
        image_split2 = image_split[0].split('_')
        output_img_dict['top_index'] = cum_height
        output_img_dict['top_md'] = image_split2[9]
        cum_height += height
        print('here')
    # depth_to_image_index = pd.Series(data=output_img_dict['top_index'], index=output_img_dict['top_md'])
    # image_index_to_depth = pd.Series(data=output_img_dict['top_md'], index=output_img_dict['top_index'])
    merged_image = np.concatenate(output_img_list)
    return merged_image#, depth_to_image_index, image_index_to_depth


def process_image(image):
    output = defaultdict(list)
    color_img = image
    output['Sum Red'] = sum(sum(color_img[:, :, 0]))
    output['Sum Green'] = sum(sum(color_img[:, :, 1]))
    output['Sum Blue'] = sum(sum(color_img[:, :, 2]))
    image = color.rgb2gray(image)
    output['Sum Luminance'] = sum(sum(image))
    edge_image = feature.canny(image, 0.2)
    v_edges = filters.sobel_v(image)
    output['Sobel V sum'] = sum(sum(v_edges))
    h_edges = filters.sobel_h(image)
    output['Sobel H sum'] = sum(sum(h_edges))
    output['max h edge count'] = np.max(np.sum(edge_image, axis=0))
    output['max v edge count'] = np.max(np.sum(edge_image, axis=1))
    gabor = filters.gabor(image, frequency=0.2)
    output['gabor filter sum'] = sum(sum(sum((gabor))))

    vert_dilation = np.array([[0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 1, 0, 0]], dtype=np.uint8)
    horz_dilation = np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [1, 1, 1, 1, 1],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]], dtype=np.uint8)
    vert_img = morphology.dilation(edge_image, vert_dilation)
    horz_img = morphology.dilation(edge_image, horz_dilation)
    segmented_img = segmentation.quickshift(color_img)

    output['Dilated Vert Image'] = np.sum(np.sum(vert_img, axis=1))
    output['Dilated Horz Image'] = np.sum(np.sum(horz_img, axis=1))
    output['Segment Count'] = len(np.unique(segmented_img))
    output_series = pd.Series(output)
    return output_series


def test_trained_models(directory="data", size=128):
    path = pkg_resources.resource_filename('core_photo_force', directory)
    las_dict = defaultdict(list)
    img_dict = defaultdict(list)
    output = defaultdict(list)
    objects = []
    with open('/Volumes/Samsung_T5/Hackathon/Force-Hackathon-Grain-Size/trained_models.pkl', 'rb') as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    models = objects[0]
    output_results = []
    for subdir, dirs, files in os.walk(path):
        print(files)
        if subdir[-len(directory):] == directory:
            for file in files:
                if file[-3:] == 'las':
                    las_dict[file[:4]] = path + '/' + file
        else:
            for file in files:
                if file[-3:] == 'jpg':
                    img_dict[file[:4]].append(subdir + '/' + file)
        counter_well = 0
        for key_well, well in img_dict.items():
            counter_well += 1
            if counter_well == 1:
                continue
            merged_image = merge_well_images(well)
            merged_image.shape[0]
            output = []
            counter = 0
            output_image = np.zeros((merged_image.shape[0], 200, 3), dtype=np.uint8)
            output_image[:, :150, :] = merged_image
            for x in range(75, merged_image.shape[0]-75, 10):
                counter += 1
                img_sample = merged_image[x- 75: x + 75, :, :]
                current_series = process_image(img_sample)
                series_frame = current_series.to_frame()
                predictions = run_trained_models_on_frame(series_frame.T, models)
                csr_mtrx = predictions[2][0]
                sed_structures_predict = csr_mtrx.toarray()
                if int(predictions[0][0]) == 0:
                    output_image[x-5:x+5, 150:200, 0] = 255
                    output_image[x - 5:x + 5, 150:200, 1] = 0
                    output_image[x - 5:x + 5, 150:200, 2] = 0
                    current_series['good_core_sample'] = False
                    current_series['sand'] = None
                    current_series['structures'] = None
                else:
                    output_image[x - 5:x + 5, 150:160, 0] = 0
                    output_image[x - 5:x + 5, 150:160, 1] = 255
                    output_image[x - 5:x + 5, 150:160, 2] = 0
                    current_series['good_core_sample'] = True
                    if int(predictions[0][0]) == 1:
                        output_image[x - 5:x + 5, 160:180, 0] = 255
                        output_image[x - 5:x + 5, 160:180, 1] = 255
                        output_image[x - 5:x + 5, 160:180, 2] = 0
                        current_series['sand'] = True
                    else:
                        output_image[x - 5:x + 5, 160:180, 0] = 0
                        output_image[x - 5:x + 5, 160:180, 1] = 0
                        output_image[x - 5:x + 5, 160:180, 2] = 255
                        current_series['sand'] = True
                    if int(sed_structures_predict[:, 0]) == 1:
                        # MASSIVE
                        output_image[x - 5:x + 5, 180:200, 0] = 255
                        output_image[x - 5:x + 5, 180:200, 1] = 255
                        output_image[x - 5:x + 5, 180:200, 2] = 0
                        current_series['structures'] = "MASSIVE"
                    elif int(sed_structures_predict[:, 1]) == 1:
                        # Laminated
                        output_image[x - 5:x + 5, 180:200, 0] = 0
                        output_image[x - 5:x + 5, 180:200, 1] = 0
                        output_image[x - 5:x + 5, 180:200, 2] = 0
                        current_series['structures'] = "LAMINATED"
                    elif int(sed_structures_predict[:, 2]) == 1:
                        # X-stratified
                        output_image[x - 5:x + 5, 180:200, 0] = 255
                        output_image[x - 5:x + 5, 180:200, 1] = 69
                        output_image[x - 5:x + 5, 180:200, 2] = 0
                        current_series['structures'] = "X-Strat"
                    elif int(sed_structures_predict[:, 3]) == 1:
                        # Burrowed
                        output_image[x - 5:x + 5, 180:200, 0] = 139
                        output_image[x - 5:x + 5, 180:200, 1] = 69
                        output_image[x - 5:x + 5, 180:200, 2] = 19
                        current_series['structures'] = "Burrowed"
                    else:
                        output_image[x - 5:x + 5, 180:200, 0] = 255
                        output_image[x - 5:x + 5, 180:200, 1] = 255
                        output_image[x - 5:x + 5, 180:200, 2] = 255
                        current_series['structures'] = "None"



                # TODO: get depth index
                output.append(current_series)
                if counter % 10 == 0:
                    print(x / output_image.shape[0])

            pred_output_dataframe = pd.concat(output, axis=1)
            output_results.append(pred_output_dataframe.T)
            path = '/Volumes/Samsung_T5/Hackathon/Force-Hackathon-Grain-Size/data/'
            with open(path + well + '.jpg', 'wb') as f:
                imsave(f, output_image, quality=50)


def run_trained_models_on_frame(frame, models_dict):
    scaler = models_dict['scaler']
    #encoder = models_dict['one_hot']
    model = models_dict['good_photo_sample_model']
    model2 = models_dict['good_photo_sample_model']
    model3 = models_dict['sed_structures']

    feature_cols = ['Dilated Horz Image', 'Dilated Vert Image', 'Segment Count',
                    'Sobel H sum', 'Sobel V sum', 'Sum Blue', 'Sum Green', 'Sum Luminance',
                    'Sum Red', 'max h edge count', 'max v edge count']
    x_table = frame[feature_cols]
    X_test = scaler.transform(x_table)
    good_core_pred = model.predict(X_test)
    sand_core_pred = model2.predict(X_test)
    sed_preds = model3.predict(X_test)

    return good_core_pred, sand_core_pred, sed_preds




