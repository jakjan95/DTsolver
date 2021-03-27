import pandas as pd
import numpy as np
import math
from src.tree_leaf import TreeLeafGeneric
from src.heurestics import gini_impurity_weighted


def heurestic_for_split(feature, split_value, samples, target_name="CLASS", heurestic=gini_impurity_weighted):
    samples_cp = samples.copy()
    # samples[feature] = pd.to_numeric(samples[feature])

    samples_cp.loc[samples[feature] >= split_value,
                   feature] = str('>= ' + str(split_value))

    samples_cp.loc[samples[feature] < split_value,
                   feature] = str('< ' + str(split_value))

    return heurestic(samples_cp, feature, target_name)


def split_continuous_variable(feature, samples, heurestic=gini_impurity_weighted):
    feature_values = list(set(samples[feature]))
    feature_values.sort()

    feature_values2 = feature_values.copy()
    feature_values2.pop(0)

    best_impurity = 1.0
    best_split = None
    zipped_values = zip(feature_values, feature_values2)

    for pair in zipped_values:
        split_value = (float(pair[0]) + float(pair[1])) / 2
        impurity = heurestic_for_split(
            feature, split_value, samples, heurestic)
        if impurity < best_impurity:
            best_impurity = impurity
            best_split = split_value

    return best_split


def is_column_numeric(data, feature):
    amount_of_values_to_be_consider_as_numeric = 2
    return data[feature].dtype != np.dtype('O') and (data[feature].dtype == np.int64 or data[feature].dtype == np.float64) and len(set(data[feature])) > amount_of_values_to_be_consider_as_numeric


def heuristic_weighted(data, feature, heurestic=gini_impurity_weighted):
    if is_column_numeric(data, feature):
        split_value = split_continuous_variable(feature, data, heurestic)
        return heurestic_for_split(feature, split_value, data, heurestic)
    else:
        return heurestic(data, feature)


def extract_values(data, feature, heuristic=gini_impurity_weighted):
    if is_column_numeric(data, feature):
        split_value = split_continuous_variable(
            feature, data, heuristic)
        values_for_given_feature = {}
        positive_item_key = ">=" + str(split_value)
        values_for_given_feature[positive_item_key] = len(
            data[data[feature] >= split_value])
        negative_item_key = "<" + str(split_value)
        values_for_given_feature[negative_item_key] = len(
            data[data[feature] < split_value])
        return values_for_given_feature
    else:
        elements, counts = np.unique(data[feature], return_counts=True)
        elements = elements.tolist()
        counts = counts.tolist()
        values_for_given_feature = {}
        for ind in range(len(elements)):
            values_for_given_feature[elements[ind]] = counts[ind]
        return values_for_given_feature


def build_tree_generic(heurestics, data, parent_node=None, is_numeric_feature=False):
    name_of_predicted_class = 'CLASS'
    if len(list(data)) == 2:
        return data[name_of_predicted_class].value_counts().idxmax()
    else:
        tree = {}
        features = list(data)
        features.remove(name_of_predicted_class)

        splitting_heurestics_values = [heuristic_weighted(
            data, given_feature, heurestics) for given_feature in features]
        best_heurestics_value_ind = np.argmin(splitting_heurestics_values)
        best_splitting_feature = features[best_heurestics_value_ind]

        if parent_node is not None:
            if splitting_heurestics_values[best_heurestics_value_ind] > parent_node.heurestic_value or splitting_heurestics_values[best_heurestics_value_ind] == 0:
                return data[name_of_predicted_class].value_counts().idxmax()

        if is_column_numeric(data, best_splitting_feature):
            root_node = TreeLeafGeneric(best_splitting_feature, extract_values(
                data, best_splitting_feature), splitting_heurestics_values[best_heurestics_value_ind], True)
        else:
            root_node = TreeLeafGeneric(best_splitting_feature, extract_values(
                data, best_splitting_feature), splitting_heurestics_values[best_heurestics_value_ind])

        tree = {root_node: {}}
        recent_best_splitting_feature = best_splitting_feature

        # numeric continuous data
        if is_column_numeric(data, recent_best_splitting_feature):
            split_value = split_continuous_variable(
                recent_best_splitting_feature, data, heurestics)

            # negative items(smaller than split)
            negative_item_key = "<" + str(split_value)
            negative_data = data[data[recent_best_splitting_feature]
                                 < split_value]
            subtree_negative = build_tree_generic(
                heurestics, negative_data, root_node, True)
            tree[root_node][negative_item_key] = subtree_negative

            # positive items(higher than split)
            positive_item_key = ">="+str(split_value)
            positive_data = data[data[recent_best_splitting_feature]
                                 >= split_value]
            subtree_positive = build_tree_generic(
                heurestics, positive_data, root_node, True)
            tree[root_node][positive_item_key] = subtree_positive

        # binary/class data
        else:
            values_for_recent_best_splitting_feature = set(
                data[recent_best_splitting_feature])

            for feature_value in values_for_recent_best_splitting_feature:
                data_with_feture_values = data[data[recent_best_splitting_feature]
                                               == feature_value]
                indexes = list(data)
                indexes.remove(recent_best_splitting_feature)
                data_with_feature_values = data_with_feture_values[indexes]
                subtree = build_tree_generic(
                    heurestics, data_with_feature_values, root_node)
                tree[root_node][feature_value] = subtree
    return tree
