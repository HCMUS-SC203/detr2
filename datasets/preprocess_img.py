import numpy as np
from PIL import Image

# a = np.array([[1, 2], [3, 4], [5, 6]])
# a = np.apply_along_axis(lambda x: x * 2, 0, a)
# print(a)

padding = 250

def activation(x, size):
    if x >= size - padding:
        return 0xFF
    # return 1
    size -= padding
    threshold = 2/5
    ratio = x/size - threshold
    # if ratio < 0:
    #     return np.exp(ratio*5) / 1.5
    # return np.tanh((ratio + 1) * 3)
    if ratio < 0:
        return 0xFF
    return 0

def process_image(img):
    img_array = np.asarray(img)
    vectorized_function = np.vectorize(activation, excluded=['size'])
    filter_vector = vectorized_function(np.arange(img_array.shape[0]), size=img_array.shape[0])
    print(filter_vector)
    # print(img_array)
    # print(img_array.shape)
    # img_array = np.apply_along_axis(lambda x: np.uint8(np.round(x * filter_vector)), 0, img_array)
    img_array = np.apply_along_axis(lambda x: np.uint8(np.bitwise_or(x, filter_vector)), 0, img_array)
    # print(img_array)
    # print(img_array.shape)
    img.close()
    img = Image.fromarray(img_array)
    return img

if __name__ == "__main__":
    img = Image.open("000029.jpg")
    img = process_image(img)
    img.show()
    img.close()