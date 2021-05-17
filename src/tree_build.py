import pandas as pd
import numpy as np
import math
from src.tree_leaf import TreeLeafGeneric
from src.heurestics import *


def heurestic_for_split(feature, split_value, samples, target_name="CLASS", heurestic=gini_impurity_weighted):
    samples_cp = samples.copy()
    # samples[feature] = pd.to_numeric(samples[feature])

    samples_cp.loc[samples[feature] >= split_value,
                   feature] = str('>= ' + str(split_value))

    samples_cp.loc[samples[feature] < split_value,
                   feature] = str('< ' + str(split_value))

    return heurestic(samples_cp, feature, target_name)


def split_continuous_variable(feature, samples, heurestic=gini_impurity_weighted):
    """
    TODO: handling different heurestics
    """
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


def create_terminal_node(data, target_name = "CLASS"):
    return data[target_name].value_counts().idxmax()


def extract_class_values_for_attributes(data, feature, heuristic=gini_impurity_weighted, target_name="CLASS"):
    attribute_values_class_count = {}
    class_values = list(set(data[target_name]))
    class_values_dict = {}
    for value in class_values:
        class_values_dict[value] = 0

    if is_column_numeric(data, feature):
        split_value = split_continuous_variable(
            feature, data, heuristic)
        positive_item_key = ">=" + str(split_value)
        negative_item_key = "<" + str(split_value)
        attribute_values_class_count[positive_item_key] = class_values_dict.copy()
        attribute_values_class_count[negative_item_key] = class_values_dict.copy()

        for i, row in data.iterrows():
            try:
                if row[feature] >= split_value:
                    attribute_values_class_count[positive_item_key][row[target_name]] += 1
                else:
                    attribute_values_class_count[negative_item_key][row[target_name]] += 1
            except KeyError:
                continue
    else:
        feature_attributes = list(set(data[feature]))
        for attribute in feature_attributes:
            attribute_values_class_count[attribute] = class_values_dict.copy()

        for i, row in data.iterrows():
            try:
                attribute_values_class_count[row[feature]][row[target_name]] += 1
            except KeyError:
                continue
    return attribute_values_class_count


data_set_size = 0
def build_tree_generic(heurestics, data, parent_node=None, is_numeric_feature=False):
    name_of_predicted_class = 'CLASS'

    global data_set_size

    if parent_node is None:
        data_set_size = len(data)

    if len(list(data)) == 2:
        return create_terminal_node(data, name_of_predicted_class)
    elif len(set(data[name_of_predicted_class])) == 1:
        """
        Class homogenity
        """
        return list(set(data[name_of_predicted_class]))[0]
    elif len(data) <= data_set_size * 0.01:
        """
        Minimum number of instance for a non-terminal node - just to avoid overgrowing
        """
        return create_terminal_node(data, name_of_predicted_class)
    else:
        tree = {}
        features = list(data)
        features.remove(name_of_predicted_class)

        splitting_heurestics_values = [heuristic_weighted(
            data, given_feature, heurestics) for given_feature in features]

        if heurestics in [info_gain, information_gain_ratio, distance_measure, j_measure, weight_of_evidence]:
            best_heurestics_value_ind = np.argmax(splitting_heurestics_values)
        elif heurestics in [gini_impurity_weighted, gini_pri, relief, relevance, mdl_simple]:
            best_heurestics_value_ind = np.argmin(splitting_heurestics_values)

        best_splitting_feature = features[best_heurestics_value_ind]

        if parent_node is not None:
            if len(set(data[best_splitting_feature])) == 1:
                """
                Attribute homogenity
                """
                return create_terminal_node(data, name_of_predicted_class)
            #Below check was to avoid overgrown trees - not needed for trees which will be pruning
            # if splitting_heurestics_values[best_heurestics_value_ind] > parent_node.heurestic_value or splitting_heurestics_values[best_heurestics_value_ind] == 0:
            #     return create_terminal_node(data, name_of_predicted_class)

        if is_column_numeric(data, best_splitting_feature):
            feature_values_extracted = extract_values(
                data, best_splitting_feature, heurestics)
            attributes_class_values_extracted = extract_class_values_for_attributes(
                data, best_splitting_feature, heurestics)
            root_node = TreeLeafGeneric(best_splitting_feature, feature_values_extracted,
                                        attributes_class_values_extracted, splitting_heurestics_values[best_heurestics_value_ind], True)
        else:
            feature_values_extracted = extract_values(
                data, best_splitting_feature, heurestics)
            attributes_class_values_extracted = extract_class_values_for_attributes(
                data, best_splitting_feature, heurestics)
            root_node = TreeLeafGeneric(best_splitting_feature, feature_values_extracted,
                                        attributes_class_values_extracted, splitting_heurestics_values[best_heurestics_value_ind])

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
