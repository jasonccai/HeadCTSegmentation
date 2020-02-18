##################################################################################
##  (c) 2020                                                                    ##
##  Radiology Informatics Lab                                                   ##
##  Department of Radiology                                                     ##
##  Mayo Clinic Rochester                                                       ##
## ---------------------------------------------------------------------------- ##
##  Source code for the manuscript entitled:                                    ##
##  "Fully automated segmentation of head CT neuroanatomy using deep learning"  ##
##  Code has been updated to Tensorflow 2                                       ##
##  For peer review only.                                                       ##
##################################################################################

import sys

print("Sort (first step): Reads Nifti files from the 'image_data' and 'mask_data' folders and sorts them to disk.")
print("Train: Trains on sorted data.")
print("Predict: Makes predictions on Nifti files in the 'image_data_predict' folder.")
flag = input("Sort, Train or Predict? (s/t/p): ")

if flag == "t":
    predict = False
    sortimg = False
    weights = ""
    testname = input("Enter a name for this training session: ")
    if "TRAIN" in testname:
        print("'TRAIN' is reserved by the script. Please choose a different filename.")
        sys.exit()
elif flag == "p":
    predict= True
    sortimg = False
    weights = input("Enter full path to model weights: ")
    slashpos = weights.find("TRAIN")
    testname = weights[slashpos+6:-14]
elif flag == "s":
    predict = False
    sortimg = True
    weights = ""
    testname = ""
else:
    print("Please enter an option.")
    exit()

import z_unetprepper as prepper
from z_unet import unet
import tensorflow as tf
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
import os
import datetime 
from tensorflow.keras.utils import get_custom_objects
from functools import partial

##################################################################################

# sets up variables
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
root = os.path.dirname(os.path.realpath(__file__))
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
if ' ' in weights:
    print("Please ensure that no spaces are present in the weights path (including the end of the path string).")
    sys.exit()

##################################################################################

# sets up some basic hyperparameters
nb_classes = 3 # number of classes (+1 for background)
TVsplit = 0.5  # training-validation split (0 to 1), splits data from the "image_data" folder volume-wise, used when sorting data
loss = "categorical_crossentropy"
optimizer = Adam(lr = 1e-4)
epochs = 10
batchsize = 3  # please reduce this if OOM error occurs

# augmentation parameters
augmentation = True
rotation = 7        # in degrees
translation_xy = 20 # in pixels
scale_xy = 0.08     # in percentage
flip_h = False      # AP flip
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

    # instantiate callbacks
    csv_logger = CSVLogger(os.path.join(savefolder, "Result.csv"))
    # lr_reducer = ReduceLROnPlateau(monitor='loss', min_delta=0.01, factor=0.5, verbose=1, cooldown=5, patience=25, min_lr=1e-10)
    best_model = ModelCheckpoint(checkpointPath, verbose=1, monitor='loss', save_best_only=True, period=epochs, mode='auto', save_weights_only=True)
    # early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=3, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    # tensorboard = TensorBoard(log_dir=os.path.join(root, 'tensorboard_logs', foldername))

    # returns Dice metric as proposed in https://arxiv.org/pdf/1606.04797.pdf
    def channel_dice(channel, y_true, y_pred):
    # channel: an interger from 0 to nb_classes (the zeroth channel being background).
        _epsilon = 10 ** -7 # prevents divide by zero.
        intersections = tf.reduce_sum(y_true[...,channel] * y_pred[...,channel])
        unions = tf.reduce_sum(y_true[...,channel] + y_pred[...,channel])
        dice_scores = (2.0 * intersections + _epsilon) / (unions + _epsilon)
        return dice_scores

    def install_channel_dicemetric(channelindex, channelname):
        customMetric = partial(channel_dice, channelindex)
        get_custom_objects().update({channelname : customMetric})
        return [channelname]
    customMetrics = install_channel_dicemetric(1,"Brain")
    customMetrics += install_channel_dicemetric(2,"CSF")

    model.compile(loss = loss, optimizer = optimizer, metrics=['accuracy'] + customMetrics, sample_weight_mode = "temporal")
 
    if TVsplit:
        model.fit_generator(generator = T, epochs = epochs, verbose = 1, shuffle = True,
                            callbacks = [csv_logger, best_model], validation_data = V,
                            max_queue_size = 10, workers = 1, use_multiprocessing = False)
    else:
        model.fit_generator(generator = T, epochs = epochs, verbose = 1, shuffle = True,
                    callbacks = [csv_logger, best_model],
                    max_queue_size = 10, workers = 1, use_multiprocessing = False)
    print("Training complete. Weights saved in", savefolder)