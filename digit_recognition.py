import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import ImageOps
import pyautogui as pg
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_image():
    im = pg.screenshot(region=(420, 200,500,500))
    im = im.resize((28,28))
    im = ImageOps.grayscale(im)
    im = ImageOps.invert(im)

    return im

model = tf.keras.models.load_model('digits_model_cnn', compile=False)

print('Prediction\t|\tAccuracy', end='')
input('')
while True:
    im = get_image()
    x = np.array(im)
    x = x.reshape((1,28,28,1))
    prediction = model.predict(x)
    print("\t{0}\t|\t{1:.4f}%".format(np.argmax(prediction), np.max(prediction)*100), end='')
    input('')

#x = x.reshape((28,28))
#plt.imshow(im, cmap=plt.cm.binary)
#plt.show()
