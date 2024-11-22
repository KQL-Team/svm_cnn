# Support Vector Machine and NasNet for Image classification

This repository contains a project that integrates Support Vector Machines (SVM) with Convolutional Neural Networks (CNN) for image classification tasks. The model is designed to classify images using SVM as the final classification layer, leveraging the power of CNN for feature extraction.
## Project Structure

```graphql
# Root directory
svm_cnn/ 
├── .idea/                # IDE configuration files (e.g., for PyCharm) 
├── Image_processing/      # Image preprocessing and dataset folder
│   ├── DATASET/           # Dataset for training and testing
│   ├── TEST/              # Test data directory
│   ├── TRAIN/             # Training data directory
├── pycache/               # Python bytecode cache 
├── nasnet_svm/            # SVM model definition and training scripts
├── train_models/          # Folder containing scripts for training models
├── .gitattributes         # Git attributes configuration file
└── README.md              # Project documentation
```
## Installation

To run this project, you need to have Python 3.x installed along with the necessary libraries. You can set up the environment by running the following command:

```bash
pip install -r requirements.txt
````
## Usage

__1.Data Preparation: Ensure your dataset is organized and ready in the Image_processing/DATASET/ folder. The dataset should be in subfolders corresponding to each class.

__2.Training the Model: You can train the model using the script nasnet_svm/nasnet_svm.ipynb. This script loads the dataset, applies CNN for feature extraction, and trains an SVM classifier for the final prediction layer.

__3.Testing the Model: After training, you can use the model for testing predictions on new images by running the test script in the experiment.ipynb.

## SVM + Nasnet Architecture
This project integrates an SVM as the final layer of a CNN model. The CNN layers are used to extract features from images, which are then passed to the SVM for classification.

The key steps in the architecture:

CNN Layers: Extract features from input images.
SVM Layer: Use the extracted features to classify images into different categories.
## Contributing
Feel free to contribute to this project by opening issues or submitting pull requests.

## License
This project is open-source and available under the MIT License.

## Acknowledgments
Keras for deep learning model implementation.
Scikit-Learn for evaluate model.
