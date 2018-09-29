w64joined_2000x260=w64joined
#Resize the image to 2000x260
w64joined_2000x260['img'] = w64joined_2000x260.apply(lambda x: cv2.resize(x['img'], (260, 2000)), axis=1)

# crop the sides of the image to 244
w64joined_2000x244=w64joined_2000x260
w64joined_2000x244['img'] = w64joined_2000x260.apply(lambda x: x['img'][:, 18:242], axis=1)