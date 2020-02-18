from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer

def NormAndActivate(data) :
    data = LayerNormalization()(data)
    return Activation("relu")(data)

def unet(nb_classes, savefolder, flag):
    
    initializer = 'he_normal'
    imageinputs = Input(shape = (512,512,1), name = "unetinput")
    
    conv1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv1A")(imageinputs)
    conv1 = NormAndActivate(conv1)
    conv1 = Conv2D(64, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv1B")(conv1)
    conv1 = NormAndActivate(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv2A")(pool1)
    conv2 = NormAndActivate(conv2)
    conv2 = Conv2D(128, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv2B")(conv2)
    conv2 = NormAndActivate(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv3A")(pool2)
    conv3 = NormAndActivate(conv3)
    conv3 = Conv2D(256, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv3B")(conv3)
    conv3 = NormAndActivate(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv4A")(pool3)
    conv4 = NormAndActivate(conv4)
    conv4 = Conv2D(512, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv4B")(conv4)
    conv4 = NormAndActivate(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(1024, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv5A")(pool4)
    conv5 = NormAndActivate(conv5)
    conv5 = Conv2D(1024, (3,3), padding = 'same', kernel_initializer = initializer, name = "conv5B")(conv5)
    conv5 = NormAndActivate(conv5)
      
    up6 = Conv2D(512, 2, padding = 'same', kernel_initializer = initializer, name = "conv6A")(UpSampling2D(size = (2,2))(conv5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = initializer, name = "conv6B")(merge6)
    conv6 = NormAndActivate(conv6)
    conv6 = Conv2D(512, 3, padding = 'same', kernel_initializer = initializer, name = "conv6C")(conv6)
    conv6 = NormAndActivate(conv6)

    up7 = Conv2D(256, 2, padding = 'same', kernel_initializer = initializer, name = "conv7A")(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = initializer, name = "conv7B")(merge7)
    conv7 = NormAndActivate(conv7)
    conv7 = Conv2D(256, 3, padding = 'same', kernel_initializer = initializer, name = "conv7C")(conv7)
    conv7 = NormAndActivate(conv7)

    up8 = Conv2D(128, 2, padding = 'same', kernel_initializer = initializer, name = "conv8A")(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = initializer, name = "conv8B")(merge8)
    conv8 = NormAndActivate(conv8)
    conv8 = Conv2D(128, 3, padding = 'same', kernel_initializer = initializer, name = "conv8C")(conv8)
    conv8 = NormAndActivate(conv8)

    up9 = Conv2D(64, 2, padding = 'same', kernel_initializer = initializer, name = "conv9A")(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = initializer, name = "conv9B")(merge9)
    conv9 = NormAndActivate(conv9)
    conv9 = Conv2D(64, 3, padding = 'same', kernel_initializer = initializer, name = "conv9C")(conv9)
    conv9 = NormAndActivate(conv9)

    conv10 = Conv2D(nb_classes, 1, activation = 'softmax', name = "conv10")(conv9)
    imageoutputs = Reshape((262144, nb_classes), name = "reshape")(conv10)
    
    model = Model(inputs = imageinputs, outputs = imageoutputs)     
   
    if not flag:
        model.summary()
        savename = savefolder + "/model"
        model_json = model.to_json()
        with open(savename + ".json", "w+") as json_file:
            json_file.write(model_json)
        plot_model(model, to_file = savename + ".png")
    
    return model