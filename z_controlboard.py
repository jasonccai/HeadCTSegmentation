##################################################################################
##  (c) 2020                                                                    ##
##  Radiology Informatics Lab                                                   ##
##  Department of Radiology                                                     ##
##  Mayo Clinic Rochester                                                       ##
## ---------------------------------------------------------------------------- ##
##  Source code for the manuscript entitled:                                    ##
##  "Fully automated segmentation of head CT neuroanatomy using deep learning"  ##
##  Code has been updated to Tensorflow 2                                       ##
##################################################################################

import os
root = os.path.dirname(os.path.realpath(__file__))
print("To use this model, please download its weights and unzip the hdf5 file into '" + root + "'.")
print("The 'image_data_predict' folder contains 3 sample test volumes. You can also copy other scans into this folder. All scans should measure 512*512 voxels axially.")
print("The model will write its predictions into 'results_folder' with a corresponding timestamp.")
input("Press enter to start or 'Ctrl+C' to exit: ")
predict= True
sortimg = False
weights = os.path.join(root, "weights.hdf5")
testname = "MODEL"
if not os.path.exists(os.path.join(root, "weights.hdf5")):
    print("Weight file not found. Please ensure that you have placed the model's weights in the correct directory.")
    raise SystemExit

### To train using your own data, please comment the section above and uncomment the section below ###

#import os
#root = os.path.dirname(os.path.realpath(__file__))
#print("Sort (first step): Reads Nifti files from the 'image_data' and 'mask_data' folders and sorts them to disk. All files should have the same filename.")
#print("Train: Trains on sorted data.")

#while True:
#    flag = input("Sort, Train, Exit? (s/t/x): ")
#    if flag == "t":
#        predict = False
#        sortimg = False
#        weights = ""     # You may enter the path to a weight file here if performing transfer learning.
#        testname = input("Enter a name for this training session (no space): ")
#        break
#    elif flag == "s":
#        predict = False
#        sortimg = True
#        weights = ""
#        testname = ""
#        break
#    elif flag == "x":
#        raise SystemExit
#    else:
#        continue
#if ' ' in testname:
#    print("Please ensure that no spaces are present in the name of the training session.")
#    raise SystemExit
#if "TRAIN" in testname:
#    print("'TRAIN' is reserved by the script. Please choose a different filename.")
#    raise SystemExit

##################################################################################

import z_unetprepper as prepper
from z_unet import unet
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import datetime 
from tensorflow.keras.utils import get_custom_objects
from functools import partial

##################################################################################

# sets up variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
images = "image_data"
labels = "mask_data"
preds = "image_data_predict"
try:
    os.mkdir(os.path.join(root, "results_folder"))
except FileExistsError:
    pass
results = "results_folder"
timenow = datetime.datetime.now().strftime('%m_%d_%H_%M_%S')
foldername = timenow + "_" + testname
savefolder = os.path.join(root, results, foldername)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
if not predict and not sortimg:
    savefolder = savefolder + "_TRAIN"
    os.mkdir(savefolder)
    checkpointPath = os.path.join(savefolder , testname + "_epoch_{epoch:02d}.hdf5")
if predict:
    savefolder = savefolder + "_PREDICT"
    os.mkdir(savefolder)

##################################################################################

# sets up some basic hyperparameters
nb_classes = 12 # number of classes (+1 for background)
TVsplit = 0.2   # training-validation split (0 to 1), splits data from the "image_data" folder volume-wise, used when sorting data
loss = "categorical_crossentropy"
optimizer = Adam(lr = 1e-4)
epochs = 500
batchsize = 3   # please reduce this if OOM error occurs

# augmentation parameters
augmentation = True
rotation = 7        # in degrees
translation_xy = 20 # in pixels
scale_xy = 0.08     # in percentage
flip_h = True       # AP flip, set to True to accommodate LAS and LPS orientations
flip_v = True       # LR flip

##################################################################################

if sortimg:
    prepper.sort(root, images, labels, TVsplit, nb_classes)
    
elif predict:
    P = prepper.Predict(nb_classes)
    model = unet(nb_classes, savefolder, predict)
    model.load_weights(weights)
    testimagepath = os.path.join(root, preds)
    for each in os.listdir(testimagepath):
        testimages = P.loadpredict(testimagepath, each)
        predlabel = model.predict(testimages, batch_size=8, verbose=1)
        P.savepredict(predlabel, savefolder, each)
    print("Prediction complete. Files saved in", savefolder + "/")

else:
    # instantiate training generator
    trainingmode = True
    T = prepper.Generate(root, trainingmode, augmentation, nb_classes, batchsize,
                         rotation, translation_xy, scale_xy, flip_h, flip_v)
    if TVsplit:
    # instantiate validation generator
        trainingmode = False
        augmentation = False
        V = prepper.Generate(root, trainingmode, augmentation, nb_classes, batchsize,
                             rotation, translation_xy, scale_xy, flip_h, flip_v)
        
    model = unet(nb_classes = nb_classes, savefolder = savefolder, flag = predict)
    if not weights=='':
        model.load_weights(weights, by_name=False)
        print("Loading weights from", weights)

    # instantiate callbacks
    csv_logger = CSVLogger(os.path.join(savefolder, "Result.csv"))
    # lr_reducer = ReduceLROnPlateau(monitor='loss', min_delta=0.01, factor=0.5, verbose=1, cooldown=5, patience=25, min_lr=1e-10)
    best_model = ModelCheckpoint(checkpointPath, verbose=1, monitor='loss', save_best_only=True, period=10, mode='auto', save_weights_only=True)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    # tensorboard = TensorBoard(log_dir=os.path.join(root, 'tensorboard_logs', foldername))

    # returns thresholded Dice                                                                # returns soft Dice (https://arxiv.org/pdf/1606.04797.pdf)
    # channel: an interger from 0 to nb_classes (the zeroth channel being background).        #
    def channel_dice(channel, y_true, y_pred):                                                # def channel_dice(channel, y_true, y_pred):
        _epsilon = 10 ** -7 # prevents divide by zero.                                        #    _epsilon = 10 ** -7
        max_prediction = tf.math.reduce_max(y_pred,axis=-1, keepdims=True)                    #    intersections = tf.reduce_sum(y_true[...,channel] * y_pred[...,channel])
        y_pred = tf.cast(tf.math.equal(y_pred, max_prediction), tf.float32)                   #    unions = tf.reduce_sum(y_true[...,channel] + y_pred[...,channel])
        intersections = tf.reduce_sum(y_true[...,channel] * y_pred[...,channel], axis=(-1))   #    dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
        unions = tf.reduce_sum(y_true[...,channel] + y_pred[...,channel], axis=(-1))          #    return dice_scores
        dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)                  #
        return tf.reduce_mean(dice_scores)                                                    #

    def install_channel_dicemetric(channelindex, channelname):
        customMetric = partial(channel_dice, channelindex)
        get_custom_objects().update({channelname : customMetric})
        return [channelname]
    customMetrics = install_channel_dicemetric(1,"Brain")
    customMetrics += install_channel_dicemetric(2,"CSF")
    customMetrics += install_channel_dicemetric(3,"DuraNSinus")
    customMetrics += install_channel_dicemetric(4,"SeptumPellucidum")
    customMetrics += install_channel_dicemetric(5,"Cerebellum")
    customMetrics += install_channel_dicemetric(6,"Caudate")
    customMetrics += install_channel_dicemetric(7,"Lentiform")
    customMetrics += install_channel_dicemetric(8,"Insular")
    customMetrics += install_channel_dicemetric(9,"InternalCapsule")
    customMetrics += install_channel_dicemetric(10,"Ventricle")
    customMetrics += install_channel_dicemetric(11,"CentralSulcus")

    model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'] + customMetrics, sample_weight_mode = "temporal")

    if TVsplit:
        model.fit_generator(generator = T, epochs = epochs, verbose = 1, shuffle = True,
                            callbacks = [csv_logger, best_model], validation_data = V,     # include other callbacks as needed
                            max_queue_size = 10, workers = 1, use_multiprocessing = False) # adjust these parameters to your hardware to maximize performance
    else:
        model.fit_generator(generator = T, epochs = epochs, verbose = 1, shuffle = True,   # this call does not perform validation (used if TVsplit set to 0)
                    callbacks = [csv_logger, best_model],                                  # include other callbacks as needed
                    max_queue_size = 10, workers = 1, use_multiprocessing = False)         # adjust these parameters to your hardware to maximize performance
    print("Training complete. Weights saved in", savefolder)
