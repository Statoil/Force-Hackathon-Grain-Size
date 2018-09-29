import lasio
if __name__ == '__main__':
    test = lasio.read('/Users/nathanieljones/PycharmProjects/Force-Hackathon-Grain-Size/data/6406_3_2 grainsize.las')
    print('test')
    test['GRAIN_SIZE']