Python code accompanying the paper "Fully Automated Segmentation of Head CT Neuroanatomy Using Deep Learning."\
Project Overview: https://jasonccai.github.io/CTBrainSegmentation/\
Paper: 

Installation Instructions:
1. Install Anaconda from:
https://www.anaconda.com/distribution/#download-section
2. (Recommended) Create a new python 3.6 environment using:\
conda create -n py36 python=3.6\
conda activate py36
3. Install Tensorflow and Nibabel using:\
pip install tensorflow-gpu==2.0.0\
pip install nibabel
4. Clone the GitHub repository to disk.
5. Download the model's weights and place them in the same folder as z_controlboard.py:\
Weights for the primary dataset only (42 normal examinations) OR\
Weights for the primary dataset and the iNPH dataset (12 examinations demonstrating ventricular enlargement) (Recommended for routine use)
5. Open a terminal and type:\
python /path/to/z_controlboard.py\
Further instructions are found in the module.

If you would like to use the model in its training state, please comment lines 12-23 and uncomment lines 27-63 in z_controlboard.py.
