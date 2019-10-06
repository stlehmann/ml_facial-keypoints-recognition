import pathlib
from typing import Tuple, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyprojroot import here


train_p: pathlib.Path = here("020_data/training.csv")
test_p: pathlib.Path = here("020_data/test.csv")


def get_labels():
    """Return available labels."""
    train_df = pd.read_csv(train_p)
    labels = list(train_df.columns)
    labels.remove("Image")
    return labels


def load_data(labels: List[str] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load train and test data."""
    assert train_p.is_file()
    assert test_p.is_file()

    # read train and test data
    train_df = pd.read_csv(train_p)
    test_df = pd.read_csv(test_p)

    # labels
    if labels is None:
        labels = list(train_df.columns)
        labels.remove("Image")
    else:
        for label in labels:
            assert label in train_df.columns

    # extract X_train, Y_train and X_test
    X_train = np.array(
        [np.array(a.split(), dtype=np.uint8) for a in train_df.Image.values],
        dtype=np.uint8,
    )
    Y_train = train_df[labels].values
    X_test = np.array(
        [np.array(a.split(), dtype=np.uint8) for a in test_df.Image.values],
        dtype=np.uint8,
    )

    return X_train, Y_train, X_test


def plot_img(img_data: Union[str, np.ndarray]) -> None:
    """Plot an image.

    The image data can be supplied as str or np.ndarray.

    """
    if isinstance(img_data, str):
        img = np.array(img_data.split(), dtype=np.uint8).reshape(96, 96)
    else:
        img = img_data.reshape(96, 96)

    plt.imshow(img, cmap="gray")
    plt.axis("off")


def plot_facial_keypoints(dataset: pd.Series) -> None:
    """Plot the whole dataset with facial keypoints"""
    plot_img(dataset.Image)
    for i in range(0, 30, 2):
        x, y = dataset[i], dataset[i + 1]
        plt.plot(x, y, "or")
