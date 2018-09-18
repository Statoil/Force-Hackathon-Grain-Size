import os

def stitch_images(directory="./data/", size=128):
    # loop through each core image folder
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            print
            os.path.join(subdir, file)
    pass

def get_grain_size_las()

class WellCore(object):
    def stitch_core_photos(self, directory):
        pass