from src.tree_leaf import TreeLeafGeneric
import src.tree_build


def predict(query, tree, default=None):
    """
    Purpose of this function is predicting class for given query(row of pandas data)
    """
    positive_key_for_numeric = 0
    negative_key_for_numeric = 1
    for tree_feature in list(tree.keys()):
        for key in list(query.keys()):
            if key == tree_feature.feature:
                try:
                    if tree_feature.is_tree_feature_numeric() == True:
                        if query[key] >= tree_feature.get_split_value():
                            positive_tree_leaf = list(tree_feature.feature_values.keys())[
                                positive_key_for_numeric]
                            result = tree[tree_feature][positive_tree_leaf]
                        else:
                            negative_tree_leaf = list(tree_feature.feature_values.keys())[
                                negative_key_for_numeric]
                            result = tree[tree_feature][negative_tree_leaf]
                    else:
                        result = tree[tree_feature][query[key]]
                except:
                    return default

                if isinstance(result, dict):
                    return predict(query, result)
                else:
                    return result


def tree_accuracy(data, tree, class_label='CLASS'):
    """
    Purpose of this function is calculating accuracy of decision tree for given testing data
    """
    queries = data.iloc[:, :-1].to_dict(orient="records")
    predicted = 0
    for i in range(len(data)):
        prediction = predict(queries[i], tree)
        if prediction == data[class_label][i]:
            predicted += 1
    return (predicted / len(data)) * 100


def number_of_levels(tree):
    if isinstance(tree, dict):
        return 1 + (max(map(number_of_levels, tree.values())) if tree else 0)
    return 0


def number_of_leafs(tree):
    """
    Purpose of this function is to calculate "rules" aka leafs
    """
    cnt = 0
    for element in tree:
        if isinstance(tree[element], dict):
            cnt += number_of_leafs(tree[element])
        else:
            cnt += 1
    return cnt
