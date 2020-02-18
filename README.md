# CTBrainSegmentation
(c) 2020
Radiology Informatics Lab\
Department of Radiology\
Mayo Clinic Rochester

Source code for the manuscript entitled: "Fully automated segmentation of head CT neuroanatomy using deep learning", uploaded for peer review.\
Two volumes and their masks are included in the "image_data" and "mask_data" folders respectively. One test volume is included in the "image_data_predict" folder.\
The model segments brain and CSF for this demonstration. Metrics do not reflect results as only one volume is used for training.

Steps:\
1. Install Anaconda from https://docs.anaconda.com/anaconda/install/
2. Install Tensorflow >=2.0 and Nibabel >= 3.0.0 in your conda environment by typing the following into your terminal:
   `pip install tensorflow` or\
   `pip install tensorflow-gpu` and\
   `pip install nibabel`
3. Run `python /path/to/folder/z_controlboard.py`
4. Enter "s" to sort the Nifti files to disk.
5. Enter "t" to train the model (takes ~5 mins on a GPU).
6. Enter "p" to perform prediction on the test volume.

If you do not have access to hardware for training and would like to see how this demo model performs on the test volume, the model's weights can be downloaded from https://drive.google.com/file/d/1xpuFGoQgGUdjejKDhBbXoj6oWfckEDcy/view?usp=sharing. Please extract this archive to the CTBrainSegmentation-master folder.


![Sample Predictions](https://github.com/jasonccai/CTBrainSegmentation/blob/master/Sample%20Performance.jpg)
