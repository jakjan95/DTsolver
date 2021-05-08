import pandas as pd
import numpy as np
import math
from decimal import *

# classic heuristics


def gini_impurity(given_items, target_name="CLASS"):
    elements, counts = np.unique(given_items[target_name], return_counts=True)
    amount_of_elements = np.sum(counts)

    if amount_of_elements == 0:
        value_of_summation = 0
    else:
        value_of_summation = np.sum(
            [Decimal((counts[i] / amount_of_elements)**2) for i in range(len(elements))])

    return Decimal(1.0) - value_of_summation


def gini_impurity_weighted(data, split_attribute_name, target_name="CLASS"):
    data_for_value = [data[data[split_attribute_name] == value]
                      for value in set(data[split_attribute_name])]

    gini_weighted = sum(Decimal(gini_impurity(
        given_data)) * Decimal(len(given_data)/len(data)) for given_data in data_for_value)
    return gini_weighted


def entropy(given_items):
    elements, counts = np.unique(given_items, return_counts=True)
    entropy = np.sum([Decimal(-counts[i]/np.sum(counts))*Decimal(np.log2(counts[i] /
                                                          np.sum(counts))) for i in range(len(elements))])
    return entropy


def info_gain(data, split_attribute_name, target_name="CLASS"):
    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    weighted_entropy = np.sum([Decimal(counts[i]/np.sum(counts))*Decimal(entropy(data.where(
        data[split_attribute_name] == vals[i]).dropna()[target_name])) for i in range(len(vals))])

    information_gain = Decimal(total_entropy) - Decimal(weighted_entropy)
    return information_gain


def information_gain_ratio(data, split_attribute_name, target_name="CLASS"):
    information_gain = info_gain(data, split_attribute_name, target_name)

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    entropy_of_feature = sum(
        [Decimal(-counts[i]/sum(counts))*Decimal(np.log2(counts[i]/sum(counts))) for i in range(len(vals))])

    if information_gain == 0 or entropy_of_feature == 0:
        information_gain_ratio = 0
    else:
        information_gain_ratio = information_gain / entropy_of_feature
    return information_gain_ratio

# Heurestics for attribute selection


def distance_measure(data, split_attribute_name, target_name="CLASS"):
    total_entropy = entropy(data[target_name])

    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    information_gain = info_gain(data, split_attribute_name, target_name)

    weighted_entropy = np.sum([Decimal(counts[i]/np.sum(counts))*Decimal(entropy(data.where(
        data[split_attribute_name] == vals[i]).dropna()[target_name])) for i in range(len(vals))])

    entropy_of_feature = np.sum(
        [Decimal(-counts[i]/np.sum(counts))*Decimal(np.log2(counts[i]/np.sum(counts))) for i in range(len(vals))])

    entropy_denominator = Decimal(weighted_entropy + entropy_of_feature)

    if entropy_denominator != 0:  # workaround for very small values
        distance = Decimal(1.0) - (information_gain / entropy_denominator)
    else:
        distance = 1

    return distance


def get_attributes_splited_and_counted(data, split_attribute_name):
    """
    Helper function - calculates number of instances from given split_attribute_name
    """
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)
    split_attribute_val_count = {}
    for i in range(len(vals)):
        split_attribute_val_count[vals[i]] = counts[i]
    return split_attribute_val_count


def get_instances_from_class_with_split_attribute_values(data, split_attribute_name, target_name="CLASS"):
    """
    Helper function - calculates number of class instances with given split attribute value
    """
    attribute_vals = np.unique(data[split_attribute_name])
    class_elements = np.unique(data[target_name])

    instances_from_class_with_split_attribute_values = {}
    for el in class_elements:
        instances_from_class_with_split_attribute_values[el] = {}
        for val in attribute_vals:
            instances_from_class_with_split_attribute_values[el][val] = 0

    for i in range(len(data)):
        recent_row = data.iloc[i]
        instances_from_class_with_split_attribute_values[recent_row[target_name]
                                                         ][recent_row[split_attribute_name]] += 1
    return instances_from_class_with_split_attribute_values


def j_measure(data, split_attribute_name, target_name="CLASS"):
    class_val_count = get_attributes_splited_and_counted(data, target_name)
    split_attribute_val_count = get_attributes_splited_and_counted(
        data, split_attribute_name)

    class_atribute_val_count = get_instances_from_class_with_split_attribute_values(
        data, split_attribute_name, target_name)
    j_measure_value = Decimal(0.0)
    attribute_probability = Decimal(0.0)
    
    for el in split_attribute_val_count:
        attribute_probability = (Decimal(
            split_attribute_val_count[el] / sum(split_attribute_val_count.values())))
        class_and_attribute_probability = Decimal(0.0)
        class_probability = Decimal(0.0)
        sum_value = Decimal(0.0)
        
        for class_el in class_atribute_val_count:
            class_and_attribute_probability = Decimal(
                class_atribute_val_count[class_el][el]/split_attribute_val_count[el])
            class_probability = Decimal(
                class_val_count[class_el]/sum(class_val_count.values()))
            
            if class_and_attribute_probability == 0:
                sum_value += Decimal(0.0)
            else:
                sum_value += Decimal(class_and_attribute_probability * \
                    Decimal(np.log2(float(class_and_attribute_probability / class_probability))))
        
        if not math.isnan(sum_value):
            j_measure_value += attribute_probability * sum_value
    
    return j_measure_value


def odds(probability):
    """
    Helper function for calculating plausibility used in Average Absoulte Weight Of Evidence
    """
    if probability != 0 and probability != 1:
        result = Decimal(probability / Decimal(1 - probability))
    else:
        result = 0
    return result


def weight_of_evidence(data, split_attribute_name, target_name="CLASS"):
    class_val_count = get_attributes_splited_and_counted(data, target_name)
    split_attribute_val_count = get_attributes_splited_and_counted(
        data, split_attribute_name)

    class_atribute_val_count = get_instances_from_class_with_split_attribute_values(
        data, split_attribute_name, target_name)

    weight_of_evidence_value = Decimal(0.0)
    class_probability = Decimal(0.0)
    for element in class_atribute_val_count:
        class_probability = Decimal(
            class_val_count[element]/sum(class_val_count.values()))
        sum_value = Decimal(0.0)
        for attribute in split_attribute_val_count:
            attribute_probability = Decimal(
                split_attribute_val_count[attribute]/sum(split_attribute_val_count.values()))
            class_and_attribute_propability = Decimal(
                class_atribute_val_count[element][attribute]/split_attribute_val_count[attribute])
            logarithm_part = 0.0
            if odds(class_and_attribute_propability) != 0 and odds(class_probability) != 0:
                logarithm_part = Decimal(np.log2(float(
                    odds(class_and_attribute_propability)/odds(class_probability))))
                    
            sum_value = attribute_probability * Decimal(abs(logarithm_part))
        weight_of_evidence_value += class_probability * sum_value
    return weight_of_evidence_value


def gini_pri(data, split_attribute_name, target_name="CLASS"):
    """
    Gini' is used in relief but it can be used alone
    """
    class_val_count = get_attributes_splited_and_counted(data, target_name)
    split_attribute_val_count = get_attributes_splited_and_counted(
        data, split_attribute_name)

    class_atribute_val_count = get_instances_from_class_with_split_attribute_values(
        data, split_attribute_name, target_name)
    sum_of_squares_attribute_probability = sum(Decimal(pow((split_attribute_val_count[attribute]/sum(
        split_attribute_val_count.values())), 2)) for attribute in split_attribute_val_count)

    equation_result = Decimal(0.0)
    first_part = Decimal(0.0)

    for attribute_element in split_attribute_val_count:
        attribute_probability = Decimal(split_attribute_val_count[attribute_element]/sum(
            split_attribute_val_count.values()))
        first_part = Decimal(pow(attribute_probability, 2) /
                             sum_of_squares_attribute_probability)
        second_part = Decimal(0.0)
        for class_element in class_atribute_val_count:
            second_part += Decimal(pow(class_atribute_val_count[class_element]
                                       [attribute_element] / split_attribute_val_count[attribute_element], 2))
        equation_result += first_part * second_part

    sum_of_square_class_probability = sum(Decimal(pow(
        class_val_count[el]/sum(class_val_count.values()), 2)) for el in class_val_count)
    gini_pri_value = equation_result - sum_of_square_class_probability
    return gini_pri_value


def relief(data, split_attribute_name, target_name="CLASS"):
    class_val_count = get_attributes_splited_and_counted(data, target_name)
    split_attribute_val_count = get_attributes_splited_and_counted(
        data, split_attribute_name)

    class_atribute_val_count = get_instances_from_class_with_split_attribute_values(
        data, split_attribute_name, target_name)

    sum_of_squares_attributes_probability = sum(Decimal(pow((split_attribute_val_count[attribute]/sum(
        split_attribute_val_count.values())), 2)) for attribute in split_attribute_val_count)
    sum_of_squares_class_probability = sum(Decimal(pow((class_val_count[class_element]/sum(
        class_val_count.values())), 2)) for class_element in class_val_count)

    gini_pri_value = gini_pri(data, split_attribute_name, target_name)

    relief_value_denominator = Decimal(
        sum_of_squares_class_probability * (Decimal(1.0) - sum_of_squares_class_probability))

    if relief_value_denominator != 0:
        relief_value = (sum_of_squares_attributes_probability *
                        gini_pri_value) / relief_value_denominator
    else:
        relief_value = 0

    return relief_value


def relevance(data, split_attribute_name, target_name="CLASS"):
    class_val_count = get_attributes_splited_and_counted(data, target_name)
    split_attribute_val_count = get_attributes_splited_and_counted(
        data, split_attribute_name)

    class_atribute_val_count = get_instances_from_class_with_split_attribute_values(
        data, split_attribute_name, target_name)

    sum_value = Decimal(0.0)
    for attribute in split_attribute_val_count:
        imj_values = []
        for class_element in class_atribute_val_count:
            recent_coefficient = Decimal(class_atribute_val_count[class_element][attribute] /
                                         class_val_count[class_element])
            imj_values.append(recent_coefficient)
        imj_index = np.argmax(imj_values)
        imj_values.pop(imj_index)
        sum_value += sum(imj_values)

    number_of_classes = len(class_val_count.keys())

    if number_of_classes == 1:
        constant_before_sum = Decimal(0.0)
    else:
        constant_before_sum = Decimal(1 / (number_of_classes - 1))

    relevance_value = Decimal(1 - constant_before_sum * sum_value)
    return relevance_value


def binominal_coefficient(upper_part, lower_part):
    """
    Binominal Coefficient used in MDL algorithm
    """
    binominal_coefficient_value = (math.factorial(
        upper_part) / (math.factorial(lower_part) * math.factorial(upper_part - lower_part)))
    return Decimal(binominal_coefficient_value)


def mdl_simple(data, split_attribute_name, target_name="CLASS"):
    """
    Simpler version of MDL - minimal description length
    """
    class_val_count = get_attributes_splited_and_counted(data, target_name)
    split_attribute_val_count = get_attributes_splited_and_counted(
        data, split_attribute_name)

    number_of_classes = len(class_val_count.keys())
    number_of_training_instances = sum(class_val_count.values())
    sum_value = Decimal(sum(math.log((binominal_coefficient(
        (split_attribute_val_count[el] + number_of_classes - 1), number_of_classes - 1)), 2) for el in split_attribute_val_count))
    first_part = Decimal(math.log(binominal_coefficient(
        number_of_training_instances + number_of_classes - 1, number_of_classes - 1)))
    information_gain_value = info_gain(data, split_attribute_name, target_name)

    mdl = Decimal(information_gain_value +
                  (Decimal(1 / number_of_training_instances))) * Decimal(first_part - sum_value)
    return mdl
