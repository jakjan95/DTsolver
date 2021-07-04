class TreeLeafGeneric:
    def __init__(self, feature, feature_values, attributes_class_values, heurestic_value, is_feature_numeric=False):
      self.feature = feature
      self.feature_values = feature_values
      self.attributes_class_values = attributes_class_values
      self.heurestic_value = heurestic_value
      self.is_feature_numeric = is_feature_numeric
      self.is_binary_categorical = False
      self.binary_left_values = None
      self.binary_right_values = None
      self.left_label=None
      self.right_label=None

    def is_tree_feature_numeric(self):
      return self.is_feature_numeric

    def get_split_value(self):
      if self.is_tree_feature_numeric():
        value_as_a_string = list(self.feature_values.keys())[0] 
        return float(value_as_a_string[2:len(value_as_a_string)])
      
    def get_string_of_feature_values(self):
      formatted_string = str()
      for key, value in self.feature_values.items():
        formatted_string += str(key) + "[" + str(value) + "]" + " "
      return formatted_string

    def __str__(self):
      return "<{0}({1:.2f})>: {2}".format(self.feature, self.heurestic_value, self.get_string_of_feature_values())

    def __repr__(self):
      return str(self)