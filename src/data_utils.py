import pandas as pd
import numpy as np


def preapare_dataset(path, labels):
    dataset = pd.read_csv(path, names=labels)
    dataset = dataset.sample(frac=1).reset_index(drop=True)  # data shuffling
    return dataset

def data_split(dataset, training_data_percentage_size = 80):
    training_size = training_data_percentage_size / 100
    end_index = int(len(dataset) * training_size)
    training_data = dataset.iloc[:end_index].reset_index(drop=True)
    testing_data = dataset.iloc[end_index:].reset_index(drop=True)
    return training_data, testing_data