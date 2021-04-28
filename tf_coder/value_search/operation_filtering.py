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
"""Utilities for operation filtering."""

import functools
import math
import operator
from typing import Any, Tuple, Type

import tensorflow as tf
from tf_coder import filter_group
from tf_coder import tensor_limits as limits
from tf_coder import tf_coder_utils
from tf_coder.value_search import value as value_module


@functools.lru_cache(maxsize=None)
def get_type_filter(desired_type):
  """Returns a value filter that only keeps values of the given type."""
  return lambda arg_value: arg_value.type is desired_type


@functools.lru_cache(maxsize=None)
def get_types_filter(desired_types: Tuple[Type[Any], ...]):
  """Returns a value filter that only keeps values with the given types."""
  return lambda arg_value: arg_value.type in desired_types


@functools.lru_cache(maxsize=None)
def get_dtype_filter(dtype):
  """Returns a value filter that only keeps tensor values of the given dtype."""
  if not isinstance(dtype, tf.DType):
    raise TypeError('dtype must be a tf.DType.')
  return lambda arg_value: arg_value.dtype is dtype


@functools.lru_cache(maxsize=None)
def get_tensor_min_rank_filter(rank):
  """Returns a value filter that only keeps tensors of high enough rank."""
  return lambda arg_value: arg_value.is_tensor and len(arg_value.shape) >= rank


def _check_tensor_finite(tensor):
  """Returns whether the float tensor contains all finite entries.

  Args:
    tensor: A float tensor. This cannot be an int tensor, or else
      tf.math.is_finite() will fail!
  """
  return bool(tf.reduce_all(tf.math.is_finite(tensor)))


def is_castable(to_cast, dtype):
  """Returns whether `to_cast` (a Value) can be safely casted to the dtype.

  This filtering strategy is a workaround for undefined behavior in TensorFlow
  (b/119633897).

  Args:
    to_cast: A Value object that would be casted.
    dtype: A Value containing a tf.DType that `to_cast` would be casted to.
  """
  if not dtype.is_int_dtype():
    return True  # We can always cast to a non-int dtype.

  to_cast_value = to_cast.value
  if to_cast.is_sparse_tensor:
    to_cast_value = to_cast.value.values

  if to_cast.is_tensor or to_cast.is_sparse_tensor:
    if not to_cast.has_float_dtype():
      return True  # Only float -> int is potentially unsafe.
    if not _check_tensor_finite(to_cast_value):
      return False  # Non-finite floats cannot be casted to int dtypes.
  elif to_cast.is_sequence:
    if to_cast.elem_type is float:
      if float('nan') in to_cast_value:
        return False  # inf and -inf will be caught by the min/max logic.
    elif to_cast.elem_type_is_tensor:
      return all(tf.size(element) and
                 is_castable(value_module.InputValue(element, 'dummy'), to_cast)
                 for element in to_cast_value)
    elif to_cast.elem_type_is_sparse_tensor:
      return all(tf.size(element.values) and
                 is_castable(value_module.InputValue(element, 'dummy'), to_cast)
                 for element in to_cast_value)
    else:
      return True  # Only lists of floats or float tensors can be unsafe.
  elif to_cast.type is float:
    if math.isnan(to_cast_value):
      return False
  else:
    return True

  min_int, max_int = tf_coder_utils.INT_DTYPE_MIN_MAX[dtype.value]

  # Floats are truncated when casted to int (nearest int in the zero direction).
  # Assuming min_int <= 0, the minimum safe float is (min_int - 1 + epsilon),
  # and the maximum safe float is (max_int + 1 - epsilon).
  return to_cast.min() > min_int - 1 and to_cast.max() < max_int + 1


def broadcastable(shape_1, shape_2):
  """Returns whether the two shapes are broadcastable."""
  return (not shape_1 or not shape_2 or
          all(x == y or x == 1 or y == 1
              for x, y in zip(shape_1[::-1], shape_2[::-1])))

# Constants for common filters. These are named with uppercase to reinforce the
# fact that these are constants and should be used as such, even though they are
# also technically functions.
# pylint: disable=invalid-name

# A filter that only keeps primitives.
PRIMITIVE_FILTER = operator.attrgetter('is_primitive')

# A filter that only keeps tf.DType objects.
DTYPE_FILTER = operator.attrgetter('is_dtype')

# A filter that only keeps sequences.
SEQUENCE_FILTER = operator.attrgetter('is_sequence')

# A filter that only keeps sequences of tensors.
TENSOR_SEQUENCE_FILTER = operator.attrgetter('elem_type_is_tensor')

# A filter that only keeps sequences of SparseTensors.
SPARSE_SEQUENCE_FILTER = operator.attrgetter('elem_type_is_sparse_tensor')

# A filter that only keeps tensors.
TENSOR_FILTER = operator.attrgetter('is_tensor')

# A filter that only keeps SparseTensors.
SPARSE_FILTER = operator.attrgetter('is_sparse_tensor')


def FLOAT_TENSOR_FILTER(arg_value):
  """Only keeps float tensors."""
  return arg_value.is_tensor and arg_value.has_float_dtype()


def NUMERIC_TENSOR_FILTER(arg_value):
  """Only keeps int and float tensors."""
  return arg_value.is_tensor and (arg_value.has_int_dtype() or
                                  arg_value.has_float_dtype())


def NONSCALAR_NUMERIC_TENSOR_FILTER(arg_value):
  """Only keeps non-scalar int and float tensors."""
  return NUMERIC_TENSOR_FILTER(arg_value) and len(arg_value.shape)


def INDICES_FILTER(arg_value):
  """Only keeps tensors/sequences containing ints suitable for indexing."""
  return ((arg_value.is_tensor and arg_value.has_int_dtype() or
           arg_value.elem_type is int) and
          arg_value.min() >= 0)


def AXIS_FILTER(arg_value):
  """Only keeps ints in the range [-1, limits.MAX_NUM_DIMENSIONS)."""
  return (arg_value.type is int and
          -1 <= arg_value.value < limits.MAX_NUM_DIMENSIONS)


def AXIS_SEQUENCE_FILTER(arg_value):
  """Only keeps sequences or 1D tensors of axis-like ints."""
  return (CONTAINS_INTS_1D_FILTER(arg_value) and
          len(arg_value.value) <= limits.MAX_NUM_DIMENSIONS and
          -1 <= arg_value.min() and
          arg_value.max() < limits.MAX_NUM_DIMENSIONS)


def PRIMITIVE_OR_SCALAR_TENSOR_FILTER(arg_value):
  """Only keeps primitives or scalar tensors."""
  return (arg_value.is_primitive or
          arg_value.is_tensor and arg_value.shape is None)


def NON_SCALAR_TENSOR_FILTER(arg_value):
  """Only keeps tensors that are not scalars."""
  return arg_value.is_tensor and arg_value.shape


def NOT_TENSOR_FILTER(arg_value):
  """Only keeps a value if it is not a Tensor or SparseTensor."""
  return not arg_value.is_tensor and not arg_value.is_sparse_tensor


def PRIMITIVE_OR_TENSOR_FILTER(arg_value):
  """Only keeps primitives and tensors."""
  return arg_value.is_primitive or arg_value.is_tensor


def TENSOR_1D_FILTER(arg_value):
  """Only keeps 1-D tensors."""
  return arg_value.is_tensor and len(arg_value.shape) == 1


def CONTAINS_INTS_FILTER(arg_value):
  """Only keeps int sequences or int tensors."""
  return arg_value.elem_type is int or arg_value.has_int_dtypes()


def CONTAINS_INTS_1D_FILTER(arg_value):
  """Only keeps int sequences or 1-D int tensors."""
  return (arg_value.elem_type is int or
          arg_value.has_int_dtype() and TENSOR_1D_FILTER(arg_value))


def TENSOR_LIKE_SEQUENCE_FILTER(arg_value):
  """Only keeps rectangular possibly-nested sequences of primitives."""
  return arg_value.is_sequence and arg_value.sequence_dtype is not None


def INT_OR_INT_TENSOR_FILTER(arg_value):
  """Only keeps int primitives or int tensors."""
  return (arg_value.type is int or
          (arg_value.is_tensor and not arg_value.shape and
           arg_value.has_int_dtype()))


def INT_LENGTH_FILTER(arg_value):
  """Only keeps int primitives or tensors representing a dimension length."""
  return (INT_OR_INT_TENSOR_FILTER(arg_value) and
          0 < int(arg_value.value) <= limits.MAX_DIMENSION_LENGTH)


def SHAPE_FILTER(arg_value):
  """Only keeps int sequences representing tensor shapes."""
  return (arg_value.elem_type is int and
          0 < len(arg_value.value) <= limits.MAX_NUM_DIMENSIONS and
          arg_value.min() > 0 and
          arg_value.max() <= limits.MAX_DIMENSION_LENGTH and
          arg_value.reduce_prod() <= limits.MAX_TENSOR_ELEMENTS)


def TENSOR_OR_SPARSE_FILTER(arg_value):
  """Only keeps Tensors and SparseTensors."""
  return arg_value.is_tensor or arg_value.is_sparse_tensor


def VECTOR_LENGTH_FILTER(arg_value):
  """Ensures that a vector of length N (N is the argument) is small enough."""
  return (INT_OR_INT_TENSOR_FILTER(arg_value) and
          0 < int(arg_value.value) <= limits.MAX_DIMENSION_LENGTH)


def SQUARE_MATRIX_SIZE_FILTER(arg_value):
  """Ensures that an NxN matrix (N is the argument) is small enough."""
  if not INT_OR_INT_TENSOR_FILTER(arg_value):
    return False
  num_rows = int(arg_value.value)
  return (0 < num_rows <= limits.MAX_DIMENSION_LENGTH and
          num_rows ** 2 <= limits.MAX_TENSOR_ELEMENTS)


def SEQUENCE_MASK_LENGTHS_FILTER(arg_value):
  """The value must contain few ints with a small maximum."""
  # Only int tensors (not SparseTensors), or list of ints, are ok.
  if not (arg_value.is_tensor and arg_value.has_int_dtype()
          or arg_value.elem_type is int):
    return False
  max_value = arg_value.max()
  num_elements = arg_value.num_elements()
  return (num_elements > 0 and
          max_value * num_elements <= limits.MAX_TENSOR_ELEMENTS)


def PADDINGS_FILTER(arg_value):
  """Must be a [N, 2] shape int32 tensor or nested sequence of ints."""
  if arg_value.is_tensor:
    dtype = arg_value.dtype
    shape = arg_value.shape
  elif arg_value.is_sequence:
    dtype = arg_value.sequence_dtype
    shape = arg_value.sequence_shape
  else:
    return False
  if not (dtype == tf.int32 and
          shape and len(shape) == 2 and shape[1] == 2 and
          shape[0] <= limits.MAX_NUM_DIMENSIONS):
    return False
  return (0 <= arg_value.min() and
          arg_value.max() < limits.MAX_DIMENSION_LENGTH / 2)


def BATCH_DIMS_FILTER(arg_value):
  """Must be an int representing a number of batch dimensions."""
  return (arg_value.type is int and
          0 <= arg_value.value < limits.MAX_NUM_DIMENSIONS)


def SCATTER_INDICES_FILTER(arg_value):
  """Must be an int tensor appropriate for indices in scatter operations."""
  return (arg_value.is_tensor and
          arg_value.has_int_dtype and
          len(arg_value.shape) >= 2 and
          arg_value.shape[-1] <= limits.MAX_NUM_DIMENSIONS and
          arg_value.min() >= 0 and
          arg_value.max() < limits.MAX_DIMENSION_LENGTH)


def SAME_DTYPES_APPLY_FILTER(arg_values):
  """Ensures that the first two arguments have the same dtype."""
  return arg_values[0].dtype == arg_values[1].dtype


def SAME_DTYPES_BROADCASTABLE_APPLY_FILTER(arg_values):
  """The two args must have the same dtypes and be broadcastable."""
  x, y = arg_values
  return x.dtype == y.dtype and broadcastable(x.shape, y.shape)


def SAME_SHAPES_APPLY_FILTER(arg_values):
  """Ensures that the first two arguments have the same shape."""
  return arg_values[0].shape == arg_values[1].shape


def TENSOR_AXIS_IN_RANGE_APPLY_FILTER(arg_values):
  """Ensures the axis is less than the rank of the tensor."""
  tensor, axis = arg_values
  return axis.value < len(tensor.shape)

# End of section for filter constants. pylint: enable=invalid-name


# LINT.IfChange(add_filters_to_function_operation)
def add_filters_to_function_operation(function_operation):
  """Adds filters to the FunctionOperation depending on its FilterGroup."""
  group = function_operation.function_info.filter_group

  if group == filter_group.FilterGroup.NONE:
    # Do nothing.
    pass

  elif group == filter_group.FilterGroup.SHAPE_1:
    function_operation.add_value_filters([SHAPE_FILTER])
  #elif group == filter_group.FilterGroup.SEQUENCE_1:
   # function_operation.add_value_filters([SEQUENCE_FILTER])
  elif group == filter_group.FilterGroup.TENSOR_1:
    function_operation.add_value_filters([TENSOR_FILTER])
  elif group == filter_group.FilterGroup.FLOATTENSOR_1:
    function_operation.add_value_filters([FLOAT_TENSOR_FILTER])
  elif group == filter_group.FilterGroup.SPARSE_1:
    function_operation.add_value_filters([SPARSE_FILTER])
  elif group == filter_group.FilterGroup.NOT_TENSOR_1:
    function_operation.add_value_filters([NOT_TENSOR_FILTER])
  elif group == filter_group.FilterGroup.PRIMITIVE_OR_TENSOR_1:
    function_operation.add_value_filters([PRIMITIVE_OR_TENSOR_FILTER])

  elif group == filter_group.FilterGroup.TENSOR_AXIS_2:
    function_operation.add_value_filters([TENSOR_FILTER, AXIS_FILTER])
    function_operation.set_apply_filter(TENSOR_AXIS_IN_RANGE_APPLY_FILTER)
  elif group == filter_group.FilterGroup.TENSOR_AXISSEQUENCE_2:
    function_operation.add_value_filters([TENSOR_FILTER, AXIS_SEQUENCE_FILTER])
    def _axis_sequence_in_range(arg_values):
      """The axes in the sequence must be in range given the tensor rank."""
      tensor, axes = arg_values
      return axes.max() < len(tensor.shape)
    function_operation.set_apply_filter(_axis_sequence_in_range)
  elif group == filter_group.FilterGroup.BOOLTENSOR_AXIS_2:
    function_operation.add_value_filters([get_dtype_filter(tf.bool),
                                          AXIS_FILTER])
    function_operation.set_apply_filter(TENSOR_AXIS_IN_RANGE_APPLY_FILTER)
  elif group == filter_group.FilterGroup.TENSORSEQUENCE_AXIS_2:
    function_operation.add_value_filters([TENSOR_SEQUENCE_FILTER,
                                          AXIS_FILTER])
  elif group == filter_group.FilterGroup.TENSOR_SHAPE_2:
    function_operation.add_value_filters([TENSOR_FILTER, SHAPE_FILTER])
  #elif group == filter_group.FilterGroup.SHAPE_PRIMITIVE_2:
   # function_operation.add_value_filters([SHAPE_FILTER, PRIMITIVE_FILTER])
  elif group == filter_group.FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2:
    function_operation.add_value_filters([NUMERIC_TENSOR_FILTER] * 2)
    function_operation.set_apply_filter(SAME_DTYPES_BROADCASTABLE_APPLY_FILTER)
  elif group == filter_group.FilterGroup.NUMERICTENSOR_AXIS_2:
    function_operation.add_value_filters([NUMERIC_TENSOR_FILTER, AXIS_FILTER])
    function_operation.set_apply_filter(TENSOR_AXIS_IN_RANGE_APPLY_FILTER)
  elif group == filter_group.FilterGroup.CASTABLE_DTYPE_2:
    function_operation.add_value_filters([DTYPE_FILTER])
    function_operation.set_apply_filter(
        lambda arg_values: is_castable(arg_values[0], arg_values[1]))
  elif group == filter_group.FilterGroup.NUMERICTENSOR_AXIS_3:
    function_operation.add_value_filters([NUMERIC_TENSOR_FILTER])
    function_operation.set_apply_filter(TENSOR_AXIS_IN_RANGE_APPLY_FILTER)
  elif group == filter_group.FilterGroup.BINCOUNT_1:
    def _bincount_filter(arg_value):
      """The value must contain nonnegative ints with a small maximum."""
      # Must be an int tensor, lists of ints, or int primitive.
      if not (arg_value.is_tensor and arg_value.has_int_dtype() or
              arg_value.elem_type is int or
              arg_value.type is int):
        return False
      max_value = arg_value.max()
      min_value = arg_value.min()
      return min_value >= 0 and max_value <= limits.MAX_DIMENSION_LENGTH
    function_operation.add_value_filters([_bincount_filter])

  elif group == filter_group.FilterGroup.EYE_1:
    function_operation.add_value_filters([SQUARE_MATRIX_SIZE_FILTER])
  elif group == filter_group.FilterGroup.NUMERICTENSOR_1:
    function_operation.add_value_filters([NUMERIC_TENSOR_FILTER])
  elif group == filter_group.FilterGroup.RANGE_1:
    function_operation.add_value_filters([VECTOR_LENGTH_FILTER])


  elif group == filter_group.FilterGroup.SQUEEZE_1:
    def _squeezable_filter(arg_value):
      return TENSOR_FILTER(arg_value) and 1 in (arg_value.shape or [])
    function_operation.add_value_filters([_squeezable_filter])

  elif group == filter_group.FilterGroup.BROADCAST_TO_2:
    def _broadcast_to_apply_filter(arg_values):
      """Checks that the tensor is broadcastable to the shape."""
      tensor, shape = arg_values
      tensor_shape = tensor.shape
      shape_values = shape.value
      if not tensor_shape or len(tensor_shape) > len(shape_values):
        return False
      return broadcastable(tensor_shape, shape_values)
    function_operation.add_value_filters([TENSOR_FILTER, SHAPE_FILTER])
    function_operation.set_apply_filter(_broadcast_to_apply_filter)

  elif group == filter_group.FilterGroup.EXPAND_DIMS_2:
    function_operation.add_value_filters([TENSOR_FILTER, AXIS_FILTER])
    def _axis_in_range(arg_values):
      """Ensures the axis is at most the rank of the tensor."""
      tensor, axis = arg_values
      return axis.value <= len(tensor.shape)
    function_operation.set_apply_filter(_axis_in_range)

  elif group == filter_group.FilterGroup.EYE_ROWS_COLS_2:
    def _eye_rows_cols_apply_filter(arg_values):
      """Checks that the result will have a small number of elements."""
      num_rows, num_cols = arg_values
      return (int(num_rows.value) * int(num_cols.value)
              <= limits.MAX_TENSOR_ELEMENTS)
    function_operation.add_value_filters([VECTOR_LENGTH_FILTER] * 2)
    function_operation.set_apply_filter(_eye_rows_cols_apply_filter)

  elif group == filter_group.FilterGroup.EYE_ROWS_DTYPE_2:
    function_operation.add_value_filters([SQUARE_MATRIX_SIZE_FILTER,
                                          DTYPE_FILTER])

  elif group == filter_group.FilterGroup.GATHER_2:
    function_operation.add_value_filters([NON_SCALAR_TENSOR_FILTER, AXIS_FILTER,
                                          INDICES_FILTER])
    def _indices_in_range(arg_values):
      params, indices = arg_values
      return (indices.max() < params.shape[0] and
              (params.num_elements() * indices.num_elements() / params.shape[0]
               <= limits.MAX_TENSOR_ELEMENTS))
    function_operation.set_apply_filter(_indices_in_range)

  
  elif group == filter_group.FilterGroup.MATMUL_2:
    def _numeric_min_rank_2_filter(arg_value):
      """Must be an int or float tensor of rank >= 2."""
      return arg_value.is_tensor and len(arg_value.shape) >= 2
    function_operation.add_value_filters([_numeric_min_rank_2_filter] * 2)
    function_operation.set_apply_filter(SAME_DTYPES_APPLY_FILTER)

  elif group == filter_group.FilterGroup.ONE_HOT_2:
    def _one_hot_indices_filter(arg_value):
      """Must contain ints and less than the max number of dimensions."""
      if arg_value.is_tensor:
        return (arg_value.has_int_dtype() and
                len(arg_value.shape) < limits.MAX_NUM_DIMENSIONS)
      else:
        return arg_value.type is int or arg_value.elem_type is int
    def _one_hot_apply_filter(arg_values):
      """Checks that the result will have a small number of elements."""
      indices, depth = arg_values
      return (indices.num_elements() * int(depth.value) <=
              limits.MAX_TENSOR_ELEMENTS)
    function_operation.add_value_filters([_one_hot_indices_filter,
                                          INT_LENGTH_FILTER])
    function_operation.set_apply_filter(_one_hot_apply_filter)

  elif group == filter_group.FilterGroup.PAD_2:
    function_operation.add_value_filters([TENSOR_FILTER, PADDINGS_FILTER])
    def _pad_2_apply_filter(arg_values):
      tensor, paddings = arg_values
      if paddings.is_tensor:
        paddings_shape = paddings.shape
      else:  # Implies paddings is a sequence.
        paddings_shape = paddings.sequence_shape
      return (tensor.shape and
              len(tensor.shape) == paddings_shape[0] and
              tf.reduce_prod(tf.add(tf.reduce_sum(paddings.value, axis=1),
                                    tensor.shape))
              <= limits.MAX_TENSOR_ELEMENTS)
    function_operation.set_apply_filter(_pad_2_apply_filter)

  elif group == filter_group.FilterGroup.SEARCHSORTED_2:
    def _sorted_last_dimension(arg_value):
      """Must be a numeric tensor that is sorted in the last dimension."""
      return (NONSCALAR_NUMERIC_TENSOR_FILTER(arg_value) and
              (arg_value.has_float_dtype() or
               arg_value.dtype in [tf.int32, tf.int64]) and
              bool(tf.reduce_all(tf.equal(arg_value.value,
                                          tf.sort(arg_value.value)))))
    function_operation.add_value_filters([_sorted_last_dimension,
                                          NONSCALAR_NUMERIC_TENSOR_FILTER])
    def _searchsorted_apply_filter(arg_values):
      """DTypes must match, dimension lengths equal except the last."""
      sorted_sequence, values = arg_values
      return (sorted_sequence.dtype == values.dtype and
              len(sorted_sequence.shape) == len(values.shape) and
              sorted_sequence.shape[:-1] == values.shape[:-1])
    function_operation.set_apply_filter(_searchsorted_apply_filter)


  elif group == filter_group.FilterGroup.TILE_2:
    def _tile_apply_filter(arg_values):
      """Checks that the result will have a small number of elements."""
      tensor, multiples = arg_values
      if multiples.is_sequence:
        dims = len(multiples.value)
      else:  # Tensor case.
        dims = multiples.shape[0]
      return (multiples.min() > 0 and
              multiples.max() > 1 and
              dims == len(tensor.shape) and
              multiples.reduce_prod() * tensor.num_elements()
              <= limits.MAX_TENSOR_ELEMENTS)
    function_operation.add_value_filters([TENSOR_FILTER,
                                          CONTAINS_INTS_1D_FILTER])
    function_operation.set_apply_filter(_tile_apply_filter)

  elif group == filter_group.FilterGroup.TOP_K_2:
    function_operation.add_value_filters([NONSCALAR_NUMERIC_TENSOR_FILTER,
                                          INT_LENGTH_FILTER])
    def _top_k_apply_filter(arg_values):
      """The tensor must have more than k items in the last dimension."""
      tensor, k = arg_values
      return int(k.value) < tensor.shape[-1]
    function_operation.set_apply_filter(_top_k_apply_filter)

  elif group == filter_group.FilterGroup.SQUEEZE_2:
    def _very_squeezable_filter(arg_value):
      """Keeps tensors with more than 1 squeezable dimension."""
      # If a tensor only has 1 squeezable dimension, then this operation is
      # useless because it is simpler to use the one-arg version of squeeze.
      return TENSOR_FILTER(arg_value) and (arg_value.shape or []).count(1) >= 2
    function_operation.add_value_filters([_very_squeezable_filter,
                                          AXIS_FILTER])
    def _squeeze_2_apply_filter(arg_values):
      tensor, axis = arg_values
      return (axis.value < len(tensor.shape) and
              tensor.shape[axis.value] == 1)
    function_operation.set_apply_filter(_squeeze_2_apply_filter)

  elif group == filter_group.FilterGroup.CLIP_BY_VALUE_3:
    function_operation.add_value_filters([NUMERIC_TENSOR_FILTER,
                                          get_types_filter((int, float)),
                                          get_types_filter((int, float))])
    def _nondecreasing_clips(arg_values):
      _, min_clip, max_clip = arg_values
      return min_clip.value <= max_clip.value
    function_operation.set_apply_filter(_nondecreasing_clips)
  elif group == filter_group.FilterGroup.PAD_3:
    function_operation.add_value_filters([TENSOR_FILTER,
                                          PADDINGS_FILTER,
                                          PRIMITIVE_OR_SCALAR_TENSOR_FILTER])
    def _pad_3_apply_filter(arg_values):
      tensor, paddings, constant_values = arg_values
      if paddings.is_tensor:
        paddings_shape = paddings.shape
      else:  # Implies paddings is a sequence.
        paddings_shape = paddings.sequence_shape
      return (tensor.shape and len(tensor.shape) == paddings_shape[0] and
              (constant_values.is_primitive or
               constant_values.dtype == tensor.dtype) and
              tf.reduce_prod(tf.add(tf.reduce_sum(paddings.value, axis=1),
                                    tensor.shape))
              <= limits.MAX_TENSOR_ELEMENTS)
    function_operation.set_apply_filter(_pad_3_apply_filter)

  elif group == filter_group.FilterGroup.RANGE_3:
    def _range_3_apply_filter(arg_values):
      """Checks that the range will end up having a small number of elements."""
      start, limit, delta = arg_values
      return (delta.value != 0 and
              0 < len(range(start.value, limit.value, delta.value))
              <= limits.MAX_DIMENSION_LENGTH)
    function_operation.add_value_filters([get_type_filter(int)] * 3)
    function_operation.set_apply_filter(_range_3_apply_filter)

  elif group == filter_group.FilterGroup.ROLL_3:
    # The case where the shift and axis are both single integers.
    function_operation.add_value_filters([TENSOR_FILTER,
                                          INT_OR_INT_TENSOR_FILTER,
                                          AXIS_FILTER])
    # The case where the shift and axis are both sequences of integers.
    function_operation.add_value_filters([TENSOR_FILTER,
                                          CONTAINS_INTS_1D_FILTER,
                                          AXIS_SEQUENCE_FILTER])
    def _roll_apply_filter(arg_values):
      tensor, shift, axis = arg_values
      if axis.type is int:
        return axis.value < len(tensor.shape)
      else:
        return (len(axis.value) == len(shift.value) and
                axis.max() < len(tensor.shape))
    function_operation.set_apply_filter(_roll_apply_filter)

  elif group == filter_group.FilterGroup.TENSORDOT_3:
    def _tensordot_arg_3_filter(arg_value):
      """The argument "axes" must have axis-like ints and the right shape."""
      if arg_value.type is int:
        # An int N means "sum over the last N axes of a and the first N axes of
        # b in order", so 0 <= N <= maximum rank.
        return 0 <= arg_value.value <= limits.MAX_NUM_DIMENSIONS
      if arg_value.elem_type is int:
        # List of length 2 is ok, elements must be valid axes.
        return (len(arg_value.value) == 2 and
                0 <= arg_value.min() and
                arg_value.max() < limits.MAX_NUM_DIMENSIONS)
      # Otherwise, must be an int tensor of shape [2] or [2, k].
      return (arg_value.is_tensor and
              arg_value.has_int_dtype() and
              1 <= len(arg_value.shape) <= 2 and
              arg_value.shape[0] == 2 and
              0 <= arg_value.min() and
              arg_value.max() < limits.MAX_NUM_DIMENSIONS)
    function_operation.add_value_filters([NONSCALAR_NUMERIC_TENSOR_FILTER,
                                          NONSCALAR_NUMERIC_TENSOR_FILTER,
                                          _tensordot_arg_3_filter])
    def _tensordot_apply_filter(arg_value):
      """First two tensors must have same dtype, and axes must be in range."""
      a, b, axes = arg_value
      if (a.dtype != b.dtype or
          # This check is overly conservative for the sake of efficiency; the
          # resulting number of elements is most likely smaller but will take
          # effort to compute more precisely.
          a.num_elements() * b.num_elements() > limits.MAX_TENSOR_ELEMENTS):
        return False
      a_rank = len(a.shape)
      b_rank = len(b.shape)
      min_rank = min(a_rank, b_rank)
      if axes.type is int:
        return axes.value <= min_rank
      elif axes.is_sequence or len(axes.shape) == 1:
        # axes is a list or tensor of shape [2].
        return axes.max() < min_rank
      else:  # axes is a tensor of shape [2, k].
        return (axes.shape[1] <= min_rank and
                tf_coder_utils.max_tensor_value(axes.value[0]) < a_rank and
                tf_coder_utils.max_tensor_value(axes.value[1]) < b_rank)

    function_operation.set_apply_filter(_tensordot_apply_filter)

  elif group == filter_group.FilterGroup.WHERE_3:
    def _where_apply_filter(arg_values):
      """Ensures that the last two arguments have matching shapes and dtypes."""
      _, x, y = arg_values
      return x.shape == y.shape and x.dtype == y.dtype
    function_operation.add_value_filters([get_dtype_filter(tf.bool),
                                          TENSOR_FILTER,
                                          TENSOR_FILTER])
    function_operation.set_apply_filter(_where_apply_filter)
  else:
    raise ValueError('Unknown filter group {} for FunctionOperation {}.'.format(
        group, function_operation.name))

# LINT.ThenChange()
# It is reasonable to strengthen or relax a filtering strategy here without
# involving a change to the filter groups.
