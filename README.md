# CTBrainSegmentation
(c) 2020
Radiology Informatics Lab\
Department of Radiology\
Mayo Clinic Rochester

Source code for the manuscript entitled: "Fully automated segmentation of head CT neuroanatomy using deep learning", uploaded for peer review.\
Two volumes and their masks are included in the "image_data" and "mask_data" folders respectively. One test volume is included in the "image_data_predict" folder.\
For demonstration, this model segments brain and CSF. Metrics do not reflect results as only one volume is used for training.

Steps:
1. Install Anaconda from https://docs.anaconda.com/anaconda/install/
2. Install Tensorflow >=2.0 and Nibabel >= 3.0.0 in your conda environment by typing the following into your terminal:
   `pip install tensorflow` or
   `pip install tensorflow-gpu` and
   `pip install nibabel`
3. Run `python /path/to/folder/z_controlboard.py`
