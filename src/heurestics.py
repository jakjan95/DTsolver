import pandas as pd
import numpy as np


def gini_impurity(given_items, target_name="CLASS"):
    elements, counts = np.unique(given_items[target_name], return_counts=True)
    amount_of_elements = np.sum(counts)

    if amount_of_elements == 0:
        value_of_summation = 0
    else:
        value_of_summation = np.sum(
            [((counts[i] / amount_of_elements)**2) for i in range(len(elements))])

    return 1.0 - value_of_summation


def gini_impurity_weighted(data, split_attribute_name, target_name="CLASS"):
    data_for_value = [data[data[split_attribute_name] == value]
                      for value in set(data[split_attribute_name])]

    gini_weighted = sum(gini_impurity(
        given_data) * (len(given_data)/len(data)) for given_data in data_for_value)

    return gini_weighted
