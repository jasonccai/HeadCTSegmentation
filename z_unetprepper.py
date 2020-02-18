import numpy as np
import nibabel as nb
import pickle
import os
import shutil
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def sort(root, images, labels, TVsplit, nb_classes):
    # compare files in "image_data" and "mask_data" folders to ensure paired data
    import filecmp
    extraimagefile = filecmp.dircmp(os.path.join(root, labels), os.path.join(root, images)).right_only
    extralabelfile = filecmp.dircmp(os.path.join(root, labels), os.path.join(root, images)).left_only
    if extraimagefile or extralabelfile:
        print("Extra image files: ",extraimagefile, "\n", "Missing image files: ", extralabelfile, "\n", "Now exiting.")
        raise SystemExit
        
    # prepare folder structure
    try:
        shutil.rmtree (os.path.join(root, "0_sorted"))
    except FileNotFoundError:
        pass

    os.mkdir(os.path.join(root, "0_sorted"))
    os.mkdir(os.path.join(root, "0_sorted", "Train"))
    os.mkdir(os.path.join(root, "0_sorted", "Validate"))

    # create data list and split into training and validation, volume-wise
    totalset = []
    for scan in os.listdir(os.path.join(root, labels)):
        totalset.append(scan)
        totalset.sort()
    splitposition = int(len(totalset)*TVsplit)
    train = totalset[splitposition:]
    validate = totalset[:splitposition]
    print("Training-Validation Split:", TVsplit, "Training:", len(train), "Validation:", len(validate))

    # pickle files to disk
    for each in train:
        print(each, "into training folder")
        trainimage = nb.load(os.path.join(root, images, each)).get_fdata()
        trainimage = trainimage.astype(np.float32)
        trainlabel = nb.load(os.path.join(root, labels, each)).get_fdata()
        trainlabel = trainlabel.astype(np.float32)

        for sliceno in range(trainlabel.shape[-1]):
            filename = each + "_" + str(sliceno+1) + ".dat"
            savefolder = os.path.join(root, "0_sorted", "Train")
            savepath = os.path.join(savefolder, filename)
            image = trainimage[:,:,sliceno:sliceno+1]
            label = trainlabel[:,:,sliceno:sliceno+1]
            dump = (image, label)
            with open (savepath, "wb+") as file:
                pickle.dump(dump, file)
                            
    for each in validate:
        print(each, "into validation folder")
        trainimage = nb.load(os.path.join(root, images, each)).get_fdata()
        trainimage = trainimage.astype(np.float32)
        trainlabel = nb.load(os.path.join(root, labels, each)).get_fdata()
        trainlabel = trainlabel.astype(np.float32)

        for sliceno in range(trainlabel.shape[-1]):
            filename = each + "_" + str(sliceno+1) + ".dat"
            savefolder = os.path.join(root, "0_sorted", "Validate")
            savepath = os.path.join(savefolder, filename)
            image = trainimage[:,:,sliceno:sliceno+1]
            label = trainlabel[:,:,sliceno:sliceno+1]
            dump = (image, label)
            with open (savepath, "wb+") as file:
                pickle.dump(dump, file)
    
    # computes class weights
    y = np.empty((512,512,0))
    for each in os.listdir(os.path.join(root, labels)):
        mask = nb.load(os.path.join(root, labels, each)).get_fdata()
        y = np.concatenate((y, mask), axis = -1)
    y = y.flatten().astype(np.uint8)
    def_clswt = y.shape / (nb_classes * np.bincount(y)) # balanced weighting from sklearn
    savepath = os.path.join(root, "0_sorted", "def_clswt.dat")
    with open (savepath, "wb+") as file:
        pickle.dump(def_clswt, file)
    att_clswt = np.cbrt(def_clswt)                      # attenuated weighting
    savepath = os.path.join(root, "0_sorted", "att_clswt.dat")
    with open (savepath, "wb+") as file:
        pickle.dump(att_clswt, file)

    print("Files sorted into '0_sorted' directory. The model can be trained now.")

class Prep(): # Reads from the "0_sorted" directory and returns a list of training and validation files
    def __init__(self, root):
        self.root = root
    
    def getdatalist(self, which):
        path = os.path.join(self.root, "0_sorted", which)
        bucket = []
        for each in os.listdir(path):
            bucket.append(os.path.join(path, each))
        return bucket
        
    def initializedatalist(self):
        train = self.getdatalist("Train")
        validate = self.getdatalist("Validate")
        return train, validate

class Generate(Sequence, Prep): # Keras generator class
    def __init__(self, root, trainingmode, augmentation, nb_classes, batchsize,
                 rotation, translation_xy, scale_xy, flip_h, flip_v):
        Prep.__init__(self, root)
        self.train, self.validate = Prep.initializedatalist(self)
        self.trainingmode = trainingmode
        self.augmentation = augmentation
        self.nb_classes = nb_classes
        self.batchsize = batchsize
        self.rotation = rotation
        self.translation_xy = translation_xy
        self.scale_xy = scale_xy
        self.flip_h = flip_h
        self.flip_v = flip_v
        self.datagenI = ImageDataGenerator(fill_mode = "constant", cval = -1000)
        self.datagenL = ImageDataGenerator(fill_mode = "constant", cval = 0)
        with open (os.path.join(root, "0_sorted", "att_clswt.dat"), "rb") as outfile:
            self.clswt = pickle.load(outfile)
            
    def wImg(self, label): # Computes the weight matrix
        wImg = np.zeros(label.shape)
        for k in range(self.nb_classes):
            ind = np.where(label == k)
            wImg[ind] = self.clswt[k]
        return wImg

    def __len__(self):
        if self.trainingmode:
            return int(np.ceil(len(self.train)/self.batchsize))
        else:
            return int(np.ceil(len(self.validate)/self.batchsize))

    def __getitem__(self, counter):
        ### TRAINING GENERATOR ###
        if self.trainingmode:
            imagebatch = np.zeros((0,512,512,1))
            labelbatch = np.zeros((0,262144,self.nb_classes))
            weightmatrixbatch = np.zeros((0,262144))
            scans = self.train[counter*self.batchsize:(counter+1)*self.batchsize]
            
            for each in scans:
                with open (each, "rb") as outfile:
                    data = pickle.load(outfile)
                    image, label = data[0], data[1]
                    image, label = image.astype(np.float32), label.astype(np.float32)
                
                weightmatrix = self.wImg(label).flatten()
                weightmatrix = np.expand_dims(weightmatrix, axis = 0)
                label = to_categorical(label, self.nb_classes)
                
                # data augmentation
                flip_h, flip_v = 0, 0
                if self.flip_h:
                    flip_h = np.random.randint(2)
                if self.flip_v:
                    flip_v = np.random.randint(2)
                parameters = {"theta": np.random.randint(-self.rotation, self.rotation+1),
                              "tx": np.random.randint(-self.translation_xy, self.translation_xy+1),
                              "ty": np.random.randint(-self.translation_xy, self.translation_xy+1),
                              "shear": 0,
                              "zx": 1 + np.random.uniform(-self.scale_xy, self.scale_xy),
                              "zy": 1 + np.random.uniform(-self.scale_xy, self.scale_xy),
                              "flip_horizontal": flip_h, "flip_vertical": flip_v}
                image, label = self.datagenI.apply_transform(image, parameters), self.datagenL.apply_transform(label, parameters)

                image, label = image.astype(np.float32), label.astype(np.float32)
                image = np.expand_dims(image, axis = 0)
                label = np.expand_dims(label, axis = 0)
                label = label.reshape(1,262144,self.nb_classes)
                imagebatch = np.concatenate((imagebatch, image), axis=0)
                labelbatch = np.concatenate((labelbatch, label), axis=0)
                weightmatrixbatch = np.concatenate((weightmatrixbatch, weightmatrix), axis=0)
            
            return (imagebatch, labelbatch, weightmatrixbatch)
        
        ### VALIDATION GENERATOR ###
        else:
            imagebatch = np.zeros((0,512,512,1))
            labelbatch = np.zeros((0,262144,self.nb_classes))
            
            scans = self.validate[counter*self.batchsize:(counter+1)*self.batchsize]
            for each in scans:
                with open (each, "rb") as outfile:
                    data = pickle.load(outfile)
                    image, label = data[0], data[1]
                    image, label = image.astype(np.float32), label.astype(np.float32)
                    
                label = to_categorical(label, self.nb_classes)
                label = label.reshape(262144,self.nb_classes)
                image = np.expand_dims(image, axis = 0)
                label = np.expand_dims(label, axis = 0)

                imagebatch = np.concatenate((imagebatch, image), axis=0)
                labelbatch = np.concatenate((labelbatch, label), axis=0)
        
            return (imagebatch, labelbatch)
    
class Predict():
    def __init__(self, nb_classes):
        self.nb_classes = nb_classes

    def loadpredict(self, testimagepath, each):
        print("Predicting:", each)
        self.testimagepath = os.path.join(testimagepath, each) 
        testimage = nb.load(self.testimagepath).get_fdata()
        self.affine = nb.load(os.path.join(testimagepath, each)).affine
        self.header = nb.load(os.path.join(testimagepath, each)).header
        self.numimgs = testimage.shape[2]
        testimage = np.moveaxis(testimage, -1, 0)
        testimage = np.expand_dims(testimage, -1).astype(np.float32)
        return testimage
    
    def savepredict(self, predlabel, savefolder, each):
        predlabel = predlabel.reshape((self.numimgs,512,512,self.nb_classes))
        predlabel = np.argmax(predlabel, axis = 3)
        predlabel = np.moveaxis(predlabel, 0, -1).astype('uint16')
        result = nb.Nifti1Image(predlabel, self.affine, self.header)
        nb.save(result, savefolder + "/" + "Prediction_" + each)
        shutil.copyfile(self.testimagepath, savefolder + "/" + "Image_" + each)
        print("Prediction complete. File saved in", savefolder + "/" + "Prediction_" + each)
        return