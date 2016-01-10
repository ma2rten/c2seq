# Sequence to Sequence

This is a C implementation of the Sequence to Sequence model described in the paper:

    Sequence to Sequence Learning with Neural Networks by Sutskever et.al, NIPS 2014.

This code is not very useful by itself. I originally planned to add CUDA and multi-GPU support. I didn't do so yet partly because my interests shifted, partly because of the release of TensorFlow which makes usage of multiple GPUs easier.

I wrote this code in C, not C++ because I am more familiar with it. However, the each of the layer modules are very "class-like" and if I were to write the code again I would write it in C++ to make of use STL vectors and to create a layer base-class.

For more information I refer to my [Python Sequence to Sequence implementation](https://github.com/ma2rten/seq2seq) which is arguably more compact.
