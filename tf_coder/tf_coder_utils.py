# Copyright 2021 The TF-Coder Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Utilities for TF-Coder."""

import functools
import itertools
import operator
from typing import List

import numpy as np
import torch #import tensorflow as tf
#from torch import tensor_limits as limits


# Types of primitive objects.
PRIMITIVE_TYPES = (int, float, bool, str)

# Int tf.DType objects for different sizes.
#UINT_DTYPES = (torch.uint32, torch.uint64, torch.uint16, torch.uint8)
INT_DTYPES = (torch.int32, torch.int64, torch.int16, torch.int8) #+ UINT_DTYPES

# Float tf.DType objects for different sizes.
FLOAT_DTYPES = (torch.float32, torch.float64, torch.float16)

# The prefix for the name of a TensorFlow function.
PY_PREFIX = 'torch.'

# Maps integral dtypes to their min and max values.
INT_DTYPE_MIN_MAX = {
    torch.int8: (-2**7, 2**7 - 1),
    torch.uint8: (0, 2**8 - 1),
    torch.int16: (-2**15, 2**15 - 1),
    torch.int32: (-2**31, 2**31 - 1),
    torch.int64: (-2**63, 2**63 - 1),
}


def get_tf_function(function_name):
  """Returns a TensorFlow function object given its name.

  Args:
    function_name: The string name of the function, e.g., "tf.matmul". Must
      start with "tf.". Nested modules are allowed, e.g., "tf.nn.softmax".

  Returns:
    The function object corresponding to function_name.

  Raises:
    ValueError: If the function name does not start with "tf.", or the function
      could not be found.
  """
  #if not function_name.startswith(PY_PREFIX):
    #raise ValueError('get_pytorch_function() called with function {}, which does '
     #                'not start with "torch.".'.format(function_name))
  function_name_without_prefix = function_name[len(PY_PREFIX):]
  try:
    tf_function = operator.attrgetter(function_name_without_prefix)(torch)
    if tf_function is None:
      raise ValueError('Could not find TF function {}'.format(function_name))
    return tf_function
  except AttributeError:
    raise ValueError('AttributeError encountered in get_tf_function for name {}'
                     .format(function_name))


def convert_to_tensor(tensor_like):
  """Converts a tensor-like object (e.g., [[1, 2], [3, 4]]) into a tf.Tensor.

  Args:
    tensor_like: A tf.Tensor, tf.SparseTensor, n-dimensional list, or a scalar.

  Returns:
    A tf.Tensor.
  """
  if isinstance(tensor_like, torch.Tensor):
    return tensor_like
  #if isinstance(tensor_like, torch.SparseTensor):
  #  return torch.sparse.reorder(tensor_like)
  return torch.tensor(tensor_like)


# rewrote
def num_tensor_elements(tensor):
  """Returns the number of elements in a tensor as an int (primitive)."""
  try: 
    print('*******', tensor)
    print(int(torch.numel(tensor)))
    return int(torch.numel(tensor))
  except:
    r = 1
    print('hi')
    for x in tensor:
        try:
            r *= x[1]
        except:
            r *= x
        
  return r


def max_tensor_value(tensor):
  """Returns the maximum value in a tensor, as a float (primitive)."""
  #return float(tensor.max(tensor.Tensor.type(torch.float32))) # old
  if type(tensor) is int: return tensor
  return float(torch.max(tensor))


def min_tensor_value(tensor):
  """Returns the minimum value in a tensor, as a float (primitive)."""
  #return float(tensor.min(tensor.Tensor.type(torch.float32))) # old
  if type(tensor) is int: return tensor
  return float(torch.min(tensor))


def tensor_to_string(tensor, decimals=5): #limits.NUM_DECIMALS
  """Converts a tensor into a string representation used for equality tests.

  TF-Coder considers two tensors to be equal if and only if their string
  representations (as computed by this function) are equal.

  Args:
    tensor: A Tensor.
    decimals: The number of floating-point decimal places to consider.

  Returns:
    A string representation of the tensor.
  """
  np_array = tensor.numpy()
  if np_array.dtype in [np.float32, np.float64]:
    np_array = np.around(np_array, decimals=decimals)

  # str(np_array.tolist()) is significantly faster than str(np_array).
  return repr(tensor.dtype) + ':' + str(np_array.tolist())


def object_to_string(obj, decimals=5): #limits.NUM_DECIMALS
  """Converts an object into a string representation used for equality tests.

  TF-Coder considers two objects to be equal if and only if their string
  representations (as computed by this function) are equal.

  Note that two sequences are considered the same if their elements are the
  same, even if one sequence is a list and the other is a tuple.

  Args:
    obj: An object, which could be a Tensor, SparseTensor, TensorFlow dtype,
      primitive (int, float, bool, or string), or sequence (list, tuple, or
      namedtuple) of other such objects.
    decimals: As described in tensor_to_string().

  Returns:
    A string representation of the object.

  Raises:
    ValueError: If `obj` has an unsupported type.
  """
  # Tensors.
  if isinstance(obj, torch.Tensor):
    return tensor_to_string(obj)
  #if isinstance(obj, torch.SparseTensor):
    # TODO(kshi): Round float SparseTensors according to `decimals`.
    #return str(obj)

  obj_type = type(obj)

  # Primitives and TensorFlow dtypes are handled the same way (with repr()).
  if obj_type in (int, float, bool, str, torch.dtype):
    if obj_type == float:
      # Floats must be rounded.
      obj = round(obj, decimals)
    return repr(obj)

  # Sequences (lists, tuples, and namedtuples) of supported objects.
  if obj_type == list or isinstance(obj, tuple):
    return 'seq[' + ', '.join(object_to_string(elem, decimals=decimals)
                              for elem in obj) + ']'

  # All other types are currently unsupported.
  raise ValueError('object_to_string called with unsupported object; type={} '
                   'and str={}.'.format(obj_type, obj))


@functools.lru_cache(maxsize=None)
def generate_partitions(num_elements: int, num_parts: int) -> List[List[int]]:
  """Generates partitions of num_elements into num_parts nonnegative parts.

  Args:
    num_elements: The number of things to permute (a nonnegative integer).
    num_parts: The number of groups to partition into (a positive integer).

  Returns:
    All possible lists of length num_parts, such that the list's elements are
    all nonnegative integers summing to num_elements.

  Raises:
    ValueError: If num_elements is negative, or num_parts is not positive.
  """
  if num_elements < 0:
    raise ValueError('In generate_partitions(), num_elements must be '
                     'nonnegative.')
  if num_parts <= 0:
    raise ValueError('In generate_partitions(), num_parts must be positive.')

  # A list [0, 1, ..., num_elements].
  choices = range(num_elements + 1)

  results = []
  # Choose (num_parts - 1) dividers among the choices, to get num_parts parts.
  for dividers in itertools.combinations_with_replacement(choices,
                                                          num_parts - 1):
    # Add dividers at the first and last choice.
    dividers = [0] + list(dividers) + [num_elements]
    # Pairwise difference between dividers gives the partition.
    results.append([next_divider - divider
                    for divider, next_divider in zip(dividers, dividers[1:])])

  return results
