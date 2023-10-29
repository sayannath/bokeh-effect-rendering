import tensorflow as tf
from tensorflow import keras

def create_model(image_height: int=1024, image_width: int=1024, num_channels: int=3) -> keras.Model:
    input_layer = keras.Input(shape=[image_height, image_width, num_channels])

    conv1 = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(input_layer)
    conv2 = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding='same', activation='relu')(conv1)
    conv3 = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=(2, 2), padding='same')(conv2)

    conv4 = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding='same')(conv2) 
    conv5 = keras.layers.Conv2D(filters=18, kernel_size=(3, 3), padding="same", activation='relu')(conv3) 

    conv6 = keras.layers.Conv2D(filters=18, kernel_size=(3, 3), strides=(2, 2), padding="same")(conv5)
    conv7 = keras.layers.Conv2D(filters=24, kernel_size=(3, 3), padding="same", activation='relu')(conv6)
    conv8 = keras.layers.Conv2D(filters=72, kernel_size=(3, 3), padding="same")(conv7)
    conv9 = keras.layers.Conv2D(filters=18, kernel_size=(3, 3), padding="same")(conv5)

    ds1 = tf.nn.depth_to_space(input=conv8, block_size=2)
    add1 = keras.layers.Add()([conv9, ds1])

    conv10 = keras.layers.Conv2D(filters=18, kernel_size=(3, 3), padding='same', activation='relu')(add1)
    conv11 = keras.layers.Conv2D(filters=48, kernel_size=(3, 3), padding='same')(conv10)

    ds2 = tf.nn.depth_to_space(input=conv11, block_size=2)
    add2 = keras.layers.Add()([conv4, ds2])

    conv12 = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same", activation='relu')(add2)
    conv13 = keras.layers.Conv2D(filters=12, kernel_size=(3, 3), padding="same", activation='relu')(conv12)

    ds3 = tf.nn.depth_to_space(input=conv13, block_size=2)

    model = keras.Model(inputs=input_layer, outputs=ds3)
    return model