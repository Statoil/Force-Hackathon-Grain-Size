from core_photo_force.scripts_modules.well_core import stitch_images
import  pickle
if __name__ == '__main__':
    objects = []
    with open('/Volumes/Samsung_T5/Hackathon/Force-Hackathon-Grain-Size/feature_table_peter.pkl', 'rb') as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    feature_table = objects[0]
    stitch_images()