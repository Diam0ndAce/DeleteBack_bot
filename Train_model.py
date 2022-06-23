import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from Preproc_data import Preproc_data
from tensorflow.keras.applications.vgg16 import VGG16
from config import save, model_savepath, image_size

# Calculate model accuracy on test dataset
def Eval(model_pred, y_test):
    y_pred = np.argmax(model_pred, axis=-1)
    y_pred = y_pred[..., np.newaxis]
    
    y_test, y_pred = y_test.reshape(-1), y_pred.reshape(-1)
    acc = accuracy_score(y_test, y_pred)
    return acc

def Initilazing_model(input_shape):
    # input and augementation layers
    input_ = layers.Input(input_shape)
    aug = layers.experimental.preprocessing.RandomContrast(0.3)(input_)
    
    # encoder (using pretrained model)
    vgg = VGG16(include_top=False, weights='imagenet', input_tensor=aug)
    vgg.trainable = False
    
    # decoder
    x = layers.UpSampling2D()(vgg.get_layer('block5_pool').output)
    x = layers.Concatenate()([x, vgg.get_layer('block5_conv3').output])
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, vgg.get_layer('block4_conv3').output])
    x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, vgg.get_layer('block3_conv3').output])
    x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, vgg.get_layer('block2_conv2').output])
    x = layers.Conv2D(16, (3,3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D()(x)
    x = layers.Concatenate()([x, vgg.get_layer('block1_conv2').output])
    
    x = layers.Conv2D(3, (3,3), padding='same', activation='sigmoid')(x)
    model = tf.keras.Model(inputs=input_, outputs=x)
    return model


if __name__ == '__main__':

    input_shape = (image_size[0], image_size[1], 3)
    
    X, y = Preproc_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = Initilazing_model(input_shape)
    print(model.summary())
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
        )
    
    batch_size = 10
    epochs = 20
    
    model.fit(X_train,
              y_train,
              batch_size=batch_size,
              epochs=epochs)
    
    model_pred = model(X_test[:100])
    acc = Eval(model_pred, y_test[:100])
    print(f'\nAccuracy of a model on test data is {round(acc, 2)}')
    
    if  save:
        model.save(model_savepath)
