import tarfile
import os
import cv2
import torch
import numpy as np

from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


def tar_caltech():
    with tarfile.open('data/caltech-101/101_ObjectCategories.tar.gz', 'r:gz') as tar:
        tar.extractall()


def data_preprocessing():
    image_paths = list(paths.list_images('101_ObjectCategories'))
    data = []
    labels = []
    label_names = []
    target_size = (224, 224)
    for image_path in image_paths:
        label = image_path.split(os.path.sep)[-2]

        # delete "background"
        if label == 'BACKGROUND_Google':
            continue

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, target_size)

        data.append(image)
        label_names.append(label)
        labels.append(label)

    data = np.array(data)
    labels = np.array(labels)

    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    count_arr = []
    label_arr = []
    for i in range(len(lb.classes_)):
        count = 0
        for j in range(len(label_names)):
            if lb.classes_[i] in label_names[j]:
                count += 1
        count_arr.append(count)
        label_arr.append(lb.classes_[i])

    # split dataset, train:validation:test = 2:1:1
    (X, x_val, Y, y_val) = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=21)
    (x_train, x_test, y_train, y_test) = train_test_split(X, Y, test_size=0.25, random_state=21)
    return x_train, y_train, x_test, y_test, x_val, y_val


class ImageDataset(Dataset):
    def __init__(self, images, labels=None, transforms=None):
        self.X = images
        self.y = torch.tensor(labels, dtype=torch.float32)
        self.transforms = transforms

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        data = self.X[item][:]

        if self.transforms:
            data = self.transforms(data)

        if self.y is not None:
            return (data, self.y[item])
        else:
            return data



if __name__ == "__main__":
    # tar_caltech()
    data_preprocessing()
