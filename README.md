## Fully Automated Segmentation of Head CT Neuroanatomy Using Deep Learning

(c) 2020, Mayo Clinic Radiology Informatics Lab\
[Project Overview](https://jasonccai.github.io/CTBrainSegmentation)

### Installation Instructions:
1. Install Anaconda from:
https://www.anaconda.com/distribution/#download-section
2. (Recommended) Create a new python 3.6 environment using:\
`conda create -n py36 python=3.6`\
`conda activate py36`
3. Install Tensorflow and Nibabel using:\
`pip install tensorflow==2.0.0`\
`pip install nibabel`
4. Clone the GitHub repository to disk.
5. Download the model's weights and place them in the same folder as `z_controlboard.py`\
[Weights](http://link.to.weights) for the training dataset only (40 normal examinations).\
OR\
[Weights](http://link.to.weights) for the primary dataset and the iNPH dataset (50 normal examinations + 12 examinations demonstrating ventricular enlargement) (Recommended for routine use).
5. Open a terminal and type:\
`python /path/to/z_controlboard.py`\
Further instructions are found in the module.

If you would like to use the model in its training state, please comment out lines 12-24 and uncomment lines 28-56 in `z_controlboard.py`. We provided 3 sample volumes in the "image_data" and "mask_data" folders for this demonstration.\
Please note that SciPy is required for the image augmentation module (`pip install scipy`).

### Alternate Installation Instructions:
• RIL-contour is a medical image annotation tool developed by our lab. It can run Tensorflow Keras models through a user interface. The instructions for downloading, installing and navigating RIL-contour are available [here](https://www.youtube.com/playlist?list=PLDlybKi3CLGibnrPIlzWInqBEgtPw1ie9). \
• The RIL-contour version of this model is available [here](http://link.to.weights). \
• A tutorial showing how to plug this model into RIL-contour is available [here](https://github.com/jasonccai/CTBrainSegmentation/blob/master/webimages/RCDemoImages/RCDemo.md).
