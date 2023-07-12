import time
import keras_cv
import matplotlib.pyplot as plt

model = keras_cv.models.StableDiffusionV2(img_width=500, img_height=500)

images = model.text_to_image("a picture of a card from the game Magic the Gathering", batch_size=4)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(2, 2, i + 1)
        plt.imshow(images[i])
        plt.axis("off")
        plt.savefig(f'fig{i}.png', bbox_inches='tight')
plot_images(images)