# importing important libraries
import  keras
import  scipy
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array,array_to_img,load_img
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# iterate over all images and augment it
for i in range (1,14):
        path=f'images/{i}.jpg'
        img = load_img(path)

        # preprocess the image
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

        # the .flow() command below generates batches of randomly transformed images
        # and saves the results to the `augmented_img/` directory
        i = 0
        for batch in datagen.flow(x, batch_size=1,
                                  save_to_dir='augmented_img', save_prefix='items', save_format='jpg'):
            i += 1
            if i > 8:
                break  # otherwise the generator would loop indefinitely