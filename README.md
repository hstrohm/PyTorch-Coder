##Base Values
TF-Coder
* Python int, float, Boolean, and string literals
* TensorFlow data types, e.g., tf.float32, tf.int64 etc.
* Variables in1, in2, etc., to reference the input tensors

becomes
* Python int, float, Boolean, and string literals
* Pytorch data types, e.g., torch.float32, torch.int64 etc.
* Variables in1, in2, etc., to reference the input tensors

##Operations
TF-Coder
*  Supported TensorFlow function calls, e.g., tf.add(x, y) and tf.math.segment_max(data, segment_ids)
* Creating a tuple from supported Python literals, e.g., (0, 1), or from other such tuples
* Various forms of indexing and slicing of sequences and tensors, e.g., tensor[-1], tensor[1:], and tensor[:, 0:5]

becomes
*  Supported PyTorch function calls, e.g., torch.add(input, other) and torch.max(input)
* Creating a tuple from supported Python literals, e.g., (0, 1), or from other such tuples
* Various forms of indexing and slicing of sequences and tensors, e.g., torch.tensor(-1), torch.tensor([1:]), and torch.tensor([:, 0:5])