import time
import keras_cv
from tensorflow import keras
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusionV2(img_width=500, img_height=500)

images = model.text_to_image("photograph of the pope evading police", batch_size=3)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.savefig(f'fig{i}.png')
plot_images(images)