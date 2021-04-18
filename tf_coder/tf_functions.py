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
"""Functions and arguments used in the TF-Coder project."""

import ast
import collections

import torch      #tensorflow -> pytorch
from tf_coder import filter_group


FilterGroup = filter_group.FilterGroup


FunctionInfo = collections.namedtuple(
    'FunctionInfo',
    ['name', 'filter_group', 'weight'])


# Weights for leaf nodes in the AST.

# Constants given by the user.
PROVIDED_CONSTANT_WEIGHT = 7

# Ubiquitous constants: 0, 1, -1.
COMMON_CONSTANT_WEIGHT = 8

# A tf.constant() wrapper around an input primitive.
PRIMITIVE_INPUT_AS_TENSOR_WEIGHT = 9

# Int constants meant to be axis values, chosen based on input tensor ranks.
AXIS_CONSTANT_WEIGHT = 14

# Int constants obtained from input/output tensor shapes.
SHAPE_CONSTANT_WEIGHT = 24

# Weight of constructing a tuple with the output shape.
OUTPUT_SHAPE_TUPLE_WEIGHT = 32

# Input variable nodes (in1, in2, etc.).
INPUT_VARIABLE_WEIGHT = 8

# DTypes with weights to add to the pool of constants.
CONSTANT_DTYPES_AND_WEIGHTS = collections.OrderedDict([
    (torch.int32, 8),
    (torch.float32, 8),
    (torch.bool, 8),
    (torch.int64, 16),
])

# Used in value search for custom cast logic.
CAST_OPERATION_NAME = 'torch.Tensor.type(dtype)'

# Used in value search to convert primitive inputs (e.g., 3) into scalar tensors
# (e.g., tf.constant(3)).
CONSTANT_OPERATION_NAME = 'tf.constant(value)'


# A list of FunctionInfo namedtuples, each describing one function usable by a
# program synthesizer. Each FunctionInfo's name contains the function name along
# with the names of the arguments for that function, in the order given in the
# function's signature. A function may appear multiple times with different
# lists of usable arguments. This list is ordered, so value search will try
# earlier functions before later ones.

# FunctionInfo name format: "tf.module.function(arg_1, arg_2, arg_3='value')"
# means call the function `tf.module.function` with varying inputs `arg_1` and
# `arg_2`, where `arg_3` is fixed and set to the literal constant `'value'`.
PY_FUNCTIONS = [
    FunctionInfo(name='torch.abs(input)',
                 filter_group=FilterGroup.PRIMITIVE_OR_TENSOR_1,
                 weight=40),
    FunctionInfo(name='torch.add(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=28),
    #FunctionInfo(name='tf.add_n(inputs)',            #couldn't find equivalent
    #             filter_group=FilterGroup.SEQUENCE_1,
    #             weight=44),
    FunctionInfo(name='torch.argmax(input, dim)', #(input, dim, keepdim=False)
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=32),
    FunctionInfo(name='torch.argmin(input, dim)', #(input, dim=None, keepdim=False)
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=48),
    FunctionInfo(name='torch.argsort(input, dim=-1, descending=False)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name=("torch.argsort(input)"),
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=48),
    FunctionInfo(name='torch.masked_fill(mask, value)',
                 filter_group=FilterGroup.TENSOR_BOOLTENSOR_2,
                 weight=28),
    FunctionInfo(name='torch.broadcast_to(input, shape)',
                 filter_group=FilterGroup.BROADCAST_TO_2,
                 weight=44),
    FunctionInfo(name=CAST_OPERATION_NAME,  # 'tf.cast(x, dtype)'.
                 filter_group=FilterGroup.CASTABLE_DTYPE_2,
                 weight=16),
    FunctionInfo(name='torch.clamp(input, min, max)',
                 filter_group=FilterGroup.CLIP_BY_VALUE_3,
                 weight=44),
    FunctionInfo(name='torch.cat(tensors, dim)',
                 filter_group=FilterGroup.TENSORSEQUENCE_AXIS_2,
                 weight=36),
    #FunctionInfo(name=CONSTANT_OPERATION_NAME,  # 'tf.constant(value)'.      #cannot find equivalent
    #             filter_group=FilterGroup.NOT_TENSOR_1,
    #             weight=23),  # Less weight than cast, accounting for the dtype.
    #FunctionInfo(name='tf.constant(value, dtype)',
    #             filter_group=FilterGroup.CASTABLE_DTYPE_2,
    #             weight=24),
    FunctionInfo(name='torch.div(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=28),
    FunctionInfo(name='torch.eq(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='torch.exp(input)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=52),
    FunctionInfo(name='torch.expand(sizes)',
                 filter_group=FilterGroup.EXPAND_DIMS_2,
                 weight=18),
    FunctionInfo(name='torch.eye(n)',
                 filter_group=FilterGroup.EYE_1,
                 weight=40),
    FunctionInfo(name='torch.eye(n,m)',
                 filter_group=FilterGroup.EYE_ROWS_COLS_2,
                 weight=60),
    FunctionInfo(name='torch.eye(n, dtype)',
                 filter_group=FilterGroup.EYE_ROWS_DTYPE_2,
                 weight=48),
    FunctionInfo(name='torch.new_full(size, fill)',   #.tensor? (obj) ^^^check if it should have torch. at the front
                 filter_group=FilterGroup.SHAPE_PRIMITIVE_2,
                 weight=40),
    FunctionInfo(name='torch.gather(input, dim, index)',
                 filter_group=FilterGroup.GATHER_2,
                 weight=24),
    #FunctionInfo(name='tf.gather(params, indices, axis, batch_dims)',
    #             filter_group=FilterGroup.GATHER_4,
    #             weight=48),
    #FunctionInfo(name='tf.gather_nd(params, indices)',
    #             filter_group=FilterGroup.GATHER_ND_2,
    #             weight=28),
    #FunctionInfo(name='tf.gather_nd(params, indices, batch_dims)',
    #             filter_group=FilterGroup.GATHER_ND_3,
    #             weight=48),
    FunctionInfo(name='torch.gt(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='torch.ge(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=32),
    FunctionInfo(name='torch.bincount(input)',
                 filter_group=FilterGroup.BINCOUNT_1,
                 weight=40),
    FunctionInfo(name='torch.ceil(input)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=44),
    FunctionInfo(name='torch.count_nonzero(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=56),
    FunctionInfo(name='torch.count_nonzero(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=56),
    FunctionInfo(name='torch.cumsum(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=44),
    #FunctionInfo(name='tf.math.cumsum(x, axis, exclusive=True)',
    #             filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
    #             weight=48),
    #FunctionInfo(name='tf.math.divide_no_nan(x, y)',
    #             filter_group=FilterGroup.SAME_DTYPE_FLOAT_BROADCASTABLE_2,
    #             weight=52),
    FunctionInfo(name='torch.floor(input)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=44),
    FunctionInfo(name='torch.log(input)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=52),
    FunctionInfo(name='torch.neg(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=48),
    FunctionInfo(name='torch.reciprocal(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=52),
    #FunctionInfo(name='tf.math.reciprocal_no_nan(x)',
    #             filter_group=FilterGroup.NUMERICTENSOR_1,
    #             weight=60),
    # FunctionInfo(name='tf.math.segment_max(data, segment_ids)',
    #              filter_group=FilterGroup.SEGMENT_OPERATION_2,
    #              weight=40),
    # FunctionInfo(name='tf.math.segment_mean(data, segment_ids)',
    #              filter_group=FilterGroup.SEGMENT_OPERATION_2,
    #              weight=56),
    # FunctionInfo(name='tf.math.segment_min(data, segment_ids)',
    #              filter_group=FilterGroup.SEGMENT_OPERATION_2,
    #              weight=48),
    # FunctionInfo(name='tf.math.segment_prod(data, segment_ids)',
    #              filter_group=FilterGroup.SEGMENT_OPERATION_2,
    #              weight=60),
    # FunctionInfo(name='tf.math.segment_sum(data, segment_ids)',
    #              filter_group=FilterGroup.SEGMENT_OPERATION_2,
    #              weight=40),
   # FunctionInfo(name='tf.math.squared_difference(x, y)',   # no equivalent (could do (input - other)**2)
    #             filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
     #            weight=52),
    FunctionInfo(name='torch.topk(input, k)',
                 filter_group=FilterGroup.TOP_K_2,
                 weight=48),
    # """FunctionInfo(name=('tf.math.unsorted_segment_max(data, segment_ids, '
    #                    'num_segments)'),
    #              filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
    #              weight=40),
    # FunctionInfo(name=('tf.math.unsorted_segment_mean(data, segment_ids, '
    #                    'num_segments)'),
    #              filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
    #              weight=56),
    # FunctionInfo(name=('tf.math.unsorted_segment_min(data, segment_ids, '
    #                    'num_segments)'),
    #              filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
    #              weight=48),
    # FunctionInfo(name=('tf.math.unsorted_segment_prod(data, segment_ids, '
    #                    'num_segments)'),
    #              filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
    #              weight=60),
    # FunctionInfo(name=('tf.math.unsorted_segment_sum(data, segment_ids, '
    #                    'num_segments)'),
    #              filter_group=FilterGroup.UNSORTED_SEGMENT_OPERATION_3,
    #              weight=40),"""
    FunctionInfo(name='torch.matmul(input, other)',
                 filter_group=FilterGroup.MATMUL_2,
                 weight=24),
    FunctionInfo(name='torch.maximum(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='torch.minimum(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=32),
    FunctionInfo(name='torch.mul(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=24),
    FunctionInfo(name='torch.ne(input, other)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=44),
    FunctionInfo(name='torch.nn.functional.one_hot(tensor)', # doesn't seem the same as tf's
                 filter_group=FilterGroup.ONE_HOT_2,
                 weight=28),
    FunctionInfo(name='torch.ones(size)',
                 filter_group=FilterGroup.SHAPE_1,
                 weight=44),
    FunctionInfo(name='torch.ones_like(input)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=36),
    # also circular mode
    FunctionInfo(name="torch.nn.functional.pad(input, pad, mode='constant')",
                 filter_group=FilterGroup.PAD_2,
                 weight=40),
    FunctionInfo(name="torch.nn.functional.pad(input, pad, mode='constant', value)",
                  filter_group=FilterGroup.PAD_3,
                  weight=52),
    FunctionInfo(name="torch.nn.functional.pad(input, pad, mode='reflect')",
                 filter_group=FilterGroup.PAD_2,
                 weight=60),
    FunctionInfo(name="torch.nn.functional.pad(input, pad, mode='replicate')",
                 filter_group=FilterGroup.PAD_2,
                 weight=60),
    FunctionInfo(name='torch.range(end)',
                 filter_group=FilterGroup.RANGE_1,
                 weight=28),
    FunctionInfo(name='torch.range(start, end, step)',
                 filter_group=FilterGroup.RANGE_3,
                 weight=56),
    # https://pytorch.org/docs/stable/torch.html#reduction-ops
    # lot more potential fcns there
    FunctionInfo(name='torch.any(input, dim)',
                 filter_group=FilterGroup.BOOLTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='torch.max(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=24),
    FunctionInfo(name='torch.max(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=24),
    FunctionInfo(name='torch.mean(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=40),
    FunctionInfo(name='torch.mean(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='torch.min(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=40),
    FunctionInfo(name='torch.min(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='torch.prod(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=52),
    FunctionInfo(name='torch.sum(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=24),
    FunctionInfo(name='torch.sum(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=24),
    FunctionInfo(name='torch.reshape(input, shape)',
                 filter_group=FilterGroup.TENSOR_SHAPE_2,
                 weight=28),
    FunctionInfo(name='torch.flip(input, dims)',
                 filter_group=FilterGroup.TENSOR_AXISSEQUENCE_2,
                 weight=48),
    FunctionInfo(name='torch.roll(input, shifts, dims)',
                 filter_group=FilterGroup.ROLL_3,
                 weight=48),
    FunctionInfo(name='torch.round(input)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=52),
   # FunctionInfo(name='tf.scatter_nd(indices, updates, shape)', # similar to scatter_add
    #             filter_group=FilterGroup.SCATTER_ND_3,
     #            weight=52),
    FunctionInfo(name="torch.searchsorted(sorted_sequence, values, right=False)",
                 filter_group=FilterGroup.SEARCHSORTED_2,
                 weight=56),
    FunctionInfo(name="torch.searchsorted(sorted_sequence, values, right=True)",
                 filter_group=FilterGroup.SEARCHSORTED_2,
                 weight=56),
    #FunctionInfo(name='tf.sequence_mask(lengths)',    # didn't find anything for these two either
     #            filter_group=FilterGroup.SEQUENCE_MASK_1,
      #           weight=32),
    #FunctionInfo(name='tf.sequence_mask(lengths, maxlen)',
     #            filter_group=FilterGroup.SEQUENCE_MASK_2,
      #           weight=44),
  #  FunctionInfo(name='torch.size(input)', # only a method fcn in pytorch
   #              filter_group=FilterGroup.TENSOR_1,
    #             weight=36),
    FunctionInfo(name='torch.sign(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=48),
    FunctionInfo(name='torch.sort(input, dim)',
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=52),
    FunctionInfo(name="torch.sort(input, dim, descending=True)",
                 filter_group=FilterGroup.NUMERICTENSOR_AXIS_2,
                 weight=60),
    FunctionInfo(name='torch.sqrt(input)',
                 filter_group=FilterGroup.FLOATTENSOR_1,
                 weight=56),
    FunctionInfo(name='torch.square(input)',
                 filter_group=FilterGroup.NUMERICTENSOR_1,
                 weight=28),
    FunctionInfo(name='torch.squeeze(input)',
                 filter_group=FilterGroup.SQUEEZE_1,
                 weight=24),
    FunctionInfo(name='torch.squeeze(input, dim)',
                 filter_group=FilterGroup.SQUEEZE_2,
                 weight=23),  # Less weight than tf.reduce_max(input, axis).
    FunctionInfo(name='torch.stack(tensors, dim)',
                 filter_group=FilterGroup.TENSORSEQUENCE_AXIS_2,
                 weight=36),
    FunctionInfo(name='torch.sub(x, y)',
                 filter_group=FilterGroup.SAME_DTYPE_NUMERIC_BROADCASTABLE_2,
                 weight=28),
   # FunctionInfo(name='torch.scatter_add(dim. index, src)',    # actually not this
    #             filter_group=FilterGroup.TENSOR_SCATTER_ND_UPDATE_3,
     #            weight=44),
    FunctionInfo(name='torch.tensordot(a, b, dims)',
                 filter_group=FilterGroup.TENSORDOT_3,
                 weight=24),
    FunctionInfo(name='torch.tile(input, reps)',
                 filter_group=FilterGroup.TILE_2,
                 weight=28),
    FunctionInfo(name='torch.transpose(input, dim0, dim1)', # 2 extra params
                 filter_group=FilterGroup.TENSOR_1,
                 weight=24),
    #FunctionInfo(name='tf.transpose(a, perm)', # should be handled by above
     #            filter_group=FilterGroup.TRANSPOSE_2,
      #           weight=44),
   # FunctionInfo(name='tf.unique_with_counts(x)',    # doesn't seem to be one, could write a custom fcn?
    #             filter_group=FilterGroup.TENSOR_1D_1,
     #            weight=48),
    FunctionInfo(name='torch.unbind(input, dim)',
                 filter_group=FilterGroup.TENSOR_AXIS_2,
                 weight=40),
    FunctionInfo(name='torch.where(condition)', # or torch.nonzeros(condition, as_tuple=True)
                 filter_group=FilterGroup.TENSOR_1,
                 weight=24),
    FunctionInfo(name='torch.where(condition, x, y)',
                 filter_group=FilterGroup.WHERE_3,
                 weight=24),
    FunctionInfo(name='torch.zeros(shape)',
                 filter_group=FilterGroup.SHAPE_1,
                 weight=40),
    FunctionInfo(name='torch.zeros_like(input)',
                 filter_group=FilterGroup.TENSOR_1,
                 weight=32),
]

# Operations for manipulating SparseTensors.
# SPARSE_FUNCTIONS = [
#     FunctionInfo(name='tf.SparseTensor(indices, values, dense_shape)',
#                  filter_group=FilterGroup.SPARSETENSOR_3,
#                  weight=20),
#     FunctionInfo(name='tf.sparse.add(a, b)',
#                  filter_group=FilterGroup.SAME_SHAPE_ONE_SPARSE_2,
#                  weight=32),
#     FunctionInfo(name='tf.sparse.concat(axis, sp_inputs)',
#                  filter_group=FilterGroup.AXIS_SPARSESEQUENCE_2,
#                  weight=40),
#     FunctionInfo(name='tf.sparse.expand_dims(sp_input, axis)',
#                  filter_group=FilterGroup.SPARSE_AXIS_2,
#                  weight=24),
#     FunctionInfo(name='tf.sparse.from_dense(tensor)',
#                  filter_group=FilterGroup.TENSOR_1,
#                  weight=20),
#     FunctionInfo(name='tf.sparse.maximum(sp_a, sp_b)',
#                  filter_group=FilterGroup.SAME_SHAPE_BOTH_SPARSE_2,
#                  weight=32),
#     FunctionInfo(name='tf.sparse.minimum(sp_a, sp_b)',
#                  filter_group=FilterGroup.SAME_SHAPE_BOTH_SPARSE_2,
#                  weight=40),
#     FunctionInfo(name='tf.sparse.reduce_max(sp_input, axis, output_is_sparse)',
#                  filter_group=FilterGroup.SPARSE_AXIS_BOOL_3,
#                  weight=28),
#     FunctionInfo(name='tf.sparse.reduce_sum(sp_input, axis, output_is_sparse)',
#                  filter_group=FilterGroup.SPARSE_AXIS_BOOL_3,
#                  weight=28),
#     FunctionInfo(name='tf.sparse.reset_shape(sp_input)',
#                  filter_group=FilterGroup.SPARSE_1,
#                  weight=40),
#     FunctionInfo(name='tf.sparse.reshape(sp_input, shape)',
#                  filter_group=FilterGroup.SPARSE_SHAPE_2,
#                  weight=40),
#     FunctionInfo(name='tf.sparse.retain(sp_input, to_retain)',
#                  filter_group=FilterGroup.SPARSE_RETAIN_2,
#                  weight=36),
#     FunctionInfo(name='tf.sparse.slice(sp_input, start, size)',
#                  filter_group=FilterGroup.SPARSE_SLICE_3,
#                  weight=32),
#     FunctionInfo(name='tf.sparse.split(sp_input, num_split, axis)',
#                  filter_group=FilterGroup.SPARSE_INT_AXIS_3,
#                  weight=32),
#     FunctionInfo(name='tf.sparse.to_dense(sp_input)',
#                  filter_group=FilterGroup.SPARSE_1,
#                  weight=20),
#     FunctionInfo(name='tf.sparse.to_dense(sp_input, default_value)',
#                  filter_group=FilterGroup.SPARSE_PRIMITIVE_2,
#                  weight=36),
#     FunctionInfo(name='tf.sparse.to_indicator(sp_input, vocab_size)',
#                  filter_group=FilterGroup.SPARSE_TO_INDICATOR_2,
#                  weight=44),
#     FunctionInfo(name='tf.sparse.transpose(sp_input)',
#                  filter_group=FilterGroup.SPARSE_1,
#                  weight=36),
#     FunctionInfo(name='tf.sparse.transpose(sp_input, perm)',
#                  filter_group=FilterGroup.SPARSE_TRANSPOSE_2,
#                  weight=56),
# ]

# A list of operation names that require filtering for value search to work at
# all, i.e., avoid segfaults and huge memory usage. Only relevant for PLDI paper
# experiments that turn off filtering for operations not listed here.
REQUIRES_FILTERING = [
    # Potentially huge memory usage.
    'torch.broadcast_to(input, shape)',
    'torch.eye(n)',
    'torch.eye(n, m)',
    'torch.eye(n, dtype)',
    'torch.new_full(size, fill)',
    'torch.bincount(input)',
    # 'tf.math.unsorted_segment_max(data, segment_ids, num_segments)',
    # 'tf.math.unsorted_segment_mean(data, segment_ids, num_segments)',
    # 'tf.math.unsorted_segment_min(data, segment_ids, num_segments)',
    # 'tf.math.unsorted_segment_prod(data, segment_ids, num_segments)',
    # 'tf.math.unsorted_segment_sum(data, segment_ids, num_segments)',
    # 'tf.one_hot(indices, depth)',
    "torch.nn.functional.pad(input, pad, mode='constant')",
    "torch.nn.functional.pad(input, pad, mode='constant', value)",
    "torch.nn.functional.pad(input, pad, mode='reflect')",
    "torch.nn.functional.pad(input, pad, mode='replicate')",
    'torch.range(end)',
    'torch.range(start, end, step)',
    # 'tf.sequence_mask(lengths)',
    # 'tf.sequence_mask(lengths, maxlen)',
    'torch.tile(input, reps)',
    'torch.zeros(shape)',

    # Avoid segfaults.
    'torch.type(dtype)',
    #'tf.constant(value, dtype)',
    'torch.gather(input, dim, index)',
    # 'tf.math.segment_max(data, segment_ids)',
    # 'tf.math.segment_mean(data, segment_ids)',
    # 'tf.math.segment_min(data, segment_ids)',
    # 'tf.math.segment_prod(data, segment_ids)',
    # 'tf.math.segment_sum(data, segment_ids)',
    'torch.reshape(input, shape)',
    'torch.squeeze(input, dim)',
    # 'tf.sparse.to_indicator(sp_input, vocab_size)',
    # 'tf.sparse.reshape(sp_input, shape)',
    # 'tf.sparse.slice(sp_input, start, size)',
    # 'tf.sparse.split(sp_input, num_split, axis)',
]


def parse_function_info_name(function_info):
  """Takes a FunctionInfo and returns (function_name, list_of_args).
  Args:
    function_info: A FunctionInfo namedtuple.
  Returns:
    A tuple (function_name, list_of_args, constant_kwargs), where function_name
    is a string, list_of_args is a list of strings, and constant_kwargs is a
    dict mapping argument names to their constant literal values. For example,
    if the FunctionInfo's name is 'tf.foo.bar(x, axis, baz=True)', then
    this function would return ('tf.foo.bar', ['x', 'axis'], {'baz': True}).
  Raises:
    ValueError: If the FunctionInfo's name is not properly formatted.
  """
  name = function_info.name

  if name.count('(') != 1:
    raise ValueError("The FunctionInfo's name must have exactly one open "
                     "parenthesis.")
  if name.count(')') != 1 or name[-1] != ')':
    raise ValueError("The FunctionInfo's name must have exactly one close "
                     "parenthesis, at the end of the name.")

  open_paren = name.index('(')
  close_paren = name.index(')')
  function_name = name[ : open_paren]
  arg_list = name[open_paren + 1 : close_paren]
  split_by_comma = [arg.strip() for arg in arg_list.split(',')]
  list_of_args = []
  constant_kwargs = collections.OrderedDict()
  for part in split_by_comma:
    if '=' in part:
      kwarg_name, literal_as_string = [x.strip() for x in part.split('=')]
      constant_kwargs[kwarg_name] = ast.literal_eval(literal_as_string)
    else:
      list_of_args.append(part)
  return function_name, list_of_args, constant_kwargs
