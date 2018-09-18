import os
import lasio
import pandas as pd
import pkg_resources
from collections import defaultdict
from skimage.io import imread, imread_collection, concatenate_images, imshow, imsave
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
        merged_image, length_of_core = merge_well_images(well)
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
    for idx, image in enumerate(well):
        print(idx / len(well))
        img = imread(image)
        width = img.shape[1]
        mid = int(width / 2)
        output_img = img[:, mid - 75:mid + 75, :]
        output_img_list.append(output_img)

        print('here')
    merged_image = np.concatenate(output_img_list)
    return merged_image, len(well)

class WellCore(object):
    def stitch_core_photos(self, directory):
        pass