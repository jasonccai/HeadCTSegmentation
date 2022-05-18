## Fully Automated Segmentation of Head CT Neuroanatomy Using Deep Learning

(c) 2020, Mayo Clinic Radiology Informatics Lab\
[Project Overview](https://jasonccai.github.io/HeadCTSegmentation)

### Installation Instructions:
1. Install Python 3.6 from:
https://www.python.org/downloads/
2. (Recommended) In a terminal, create a new Python environment using venv:\
`python3 -m venv py36`\
`source py36/bin/activate`\
`pip install -U pip`
3. Install Tensorflow and Nibabel into the newly-created "py36" environment using:\
`pip install tensorflow==2.2.0`\
`pip install nibabel`
4. Clone the GitHub repository to disk.
5. Download the model's weights and place them in the same folder as `z_controlboard.py`\
[Weights](https://drive.google.com/file/d/1h-mS_JywoBotS_qT88ob0qfzYfEqu0vS/view?usp=sharing) for the training dataset only (40 normal examinations).\
OR\
[Weights](https://drive.google.com/file/d/1fSe7oDaE8NYh5GLUN6h-Yo-ZlrJHXeuZ/view?usp=sharing) for the primary dataset and the iNPH dataset (50 normal examinations + 12 examinations demonstrating ventricular enlargement; recommended for routine use).
5. Open a terminal and type:\
`python /path/to/z_controlboard.py`\
Further instructions are found in the module.

If you would like to use the model in its training state, please comment out lines 12-24 and uncomment lines 28-56 in `z_controlboard.py`. We provided 3 sample volumes in the "image_data" and "mask_data" folders for this demonstration.\
Please note that SciPy is required for the image augmentation module (`pip install scipy`).

### Alternate Installation Instructions:
• RIL-Contour is a medical image annotation tool developed by our lab. It can run Tensorflow Keras models through a user interface. The instructions for downloading, installing and navigating RIL-Contour are available [here](https://www.youtube.com/playlist?list=PLDlybKi3CLGibnrPIlzWInqBEgtPw1ie9). \
• A tutorial showing how to run our model in RIL-Contour is available [here](https://github.com/jasonccai/CTBrainSegmentation/blob/master/webimages/RCDemoImages/RCDemo.md).

### Citation:
JC Cai, Z Akkus, KA Philbrick, A Boonrod, S Hoodeshenas, AD Weston, P Rouzrokh, GM Conte, A Zeinoddini, DC Vogelsang, Q Huang, BJ Erickson\
*“Fully Automated Segmentation of Neuroanatomy on Head CT Using Deep Learning”*\
Radiol Artif Intell. 2020 Sep; 2(5):e190183. [https://doi.org/10.1148/ryai.2020190183](https://doi.org/10.1148/ryai.2020190183)\
Click [here](https://pubs.rsna.org/action/showCitFormats?doi=10.1148%2Fryai.2020190183) to download citation data.\
\
For inquires, please email jason.cai outlook com
