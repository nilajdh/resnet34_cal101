# Caltech-101 Image Classification with ResNet34

## Project Overview
This project aims to perform image classification on the Caltech-101 dataset using a pre-trained ResNet34 model. The project includes steps such as data preprocessing, model training, and model testing. TensorBoard is used to record the loss and accuracy during the training process.

## Environment Requirements
- Python 3.9
- PyTorch
- OpenCV
- Scikit-learn
- imutils
- TensorBoard

You can install the required libraries using the following command:
```bash
pip install torch torchvision opencv-python scikit-learn imutils tensorboard
```

## Dataset
This project uses the Caltech-101 dataset. You need to download and extract the dataset to the `data/caltech-101` directory, ensuring that the dataset file is named `101_ObjectCategories.tar.gz`.

## File Structure
```
.
├── train.py                # Model training script
├── test.py                 # Model testing script
├── data_pre.py             # Data preprocessing script
├── logs                    # Directory to save training logs
├── test_results            # Directory to save test results
├── 101_ObjectCategories    # Unzipped dataset
└── data                    # Downloaded dataset archive
    └── caltech-101
        └── 101_ObjectCategories.tar.gz  # Dataset file
```

## Usage

### Data Preprocessing
Run the `data_pre.py` script to unzip downloaded dataset. This file This python file defines the data processing functions that will be called later.
```bash
python data_pre.py
```

### Model Training
Run the `train.py` script to train the model. The loss and accuracy during the training process will be recorded in TensorBoard.
You can change the hyperparameters in this file, including batch_size, epochs, pretrained(True\False), lr_base and lr_fc ...
```bash
python train.py
```

### Model Testing
Run the `test.py` script to test the model. You need to specify the path to the model weight file.
```bash
python test.py --model_path .logsresnet34_finetune_60epoch_pretrained_checkpoint.pth --num_classes 101 --batch_size 16 --results_dir .test_results
```
Parameter Description:
- `--model_path`: Path to the model weight file, required.
- `--num_classes`: Number of classification categories, default is 101.
- `--batch_size`: Batch size, default is 16.
- `--results_dir`: Directory to save test results, default is `./test_results`.

## Viewing Training Logs
Use TensorBoard to view the loss and accuracy during the training process:
```bash
tensorboard --logdir=./logs
```
Then open `http://localhost:6006` in your browser to view the logs.

## Test Results
The test results will be saved in the `classification_report.txt` file in the `test_results` directory, including the classification report and overall metrics (accuracy, precision, recall, F1-score).

## Notes
- Please ensure that the dataset file path is correct and the dataset file exists.
- In the train file set the hyperparameters you want to train.