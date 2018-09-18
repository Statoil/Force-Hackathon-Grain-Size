from unittest import TestCase
from .well_core import WellCore
import pkg_resources


class TestWellCore(TestCase):
    def test_stitch_core_photos(self):

        obj = WellCore()
        path = pkg_resources.resource_filename('core_photo_force', 'well_6406_3_2_images')
        combined_image = obj.stitch_core_photos(
            directory='/Users/nathanieljones/PycharmProjects/Force-Hackathon-Grain-Size/data/well_6406_3_2_images')
