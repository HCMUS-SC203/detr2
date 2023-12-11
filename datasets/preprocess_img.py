import numpy as np
from PIL import Image

# a = np.array([[1, 2], [3, 4], [5, 6]])
# a = np.apply_along_axis(lambda x: x * 2, 0, a)
# print(a)

padding = 250

def activation(x, size):
    if x >= size - padding:
        return 0
    size -= padding
    ratio = x/size - 0.5
    if ratio < 0:
        return np.exp(ratio*10) / 1.5
    return (ratio + 1) / 1.5

def process_image(img):
    img_array = np.asarray(img)
    vectorized_function = np.vectorize(activation, excluded=['size'])
    filter_vector = vectorized_function(np.arange(img_array.shape[0]), size=img_array.shape[0])
    # print(filter_vector)
    # print(img_array)
    # print(img_array.shape)
    img_array = np.apply_along_axis(lambda x: np.uint8(np.round(x * filter_vector)), 0, img_array)
    # print(img_array)
    # print(img_array.shape)
    img.close()
    img = Image.fromarray(img_array)
    return img

# img = Image.open("000020.jpg")
# img = process_image(img)
# img.show()
# img.close()