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
        merged_image, depth_to_image_index, image_index_to_depth = merge_well_images(well)
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
    depth_to_image_index = pd.Series(data=output_img_dict['top_index'], index=output_img_dict['top_md'])
    image_index_to_depth = pd.Series(data=output_img_dict['top_md'], index=output_img_dict['top_index'])
    merged_image = np.concatenate(output_img_list)
    return merged_image, depth_to_image_index, image_index_to_depth


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