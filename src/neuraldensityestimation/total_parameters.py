
def total_parameters(trainable_variables):
  total_parameters = 0
  for variable in trainable_variables:
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      #print(shape)
      #print(len(shape))
      variable_parameters = 1
      for dim in shape:
          #print(dim)
          variable_parameters *= dim
      #print(variable_parameters)
      total_parameters += variable_parameters
  return total_parameters


