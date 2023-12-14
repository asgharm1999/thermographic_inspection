# Segformer Architecture

## Introduction

A PyTorch implementation of [Segformer](https://arxiv.org/abs/2105.15203), a transformer-based model with a MLP decoder for semantic segmentation, was used. The model was pulled from Huggingface, an online transformer repository, and the mit-b0 configuration and weights were used.

## Architecture

This is a quick summary of the architecture of the model. For more details, please refer to the [paper](https://arxiv.org/abs/2105.15203).

### Input

The model creates overlapped patch embeddings of size 4x4, as opposed to ViT's 16x16. The smaller patches favors dense prediction.

The model will not interleave positional embeddings, if the images during inference are of different sizes than the ones used during training. This should allow the model to better adapt to different image resolutions.

The patch embeddings will then be pushed through a series of transformer blocks. 

### Transformer Blocks

Each block will have an efficient self-attention block, that is designed to improve upon the original self-attention block. Originally, the block would take in three heads K, Q, and V, each of dimensions N x C, where N = H x W. Computing self-attention on these heads is computationally expensive, so efficient self-attention will reshape the heads, and push them through a linear layer. This will reduce the size of the heads.

The transformer block will also have a Mix-FFN block, which applies an MLP and 3x3 convolution to each FFN to add positional information. 

Finally, the transformer block will use patch merging, which will merge the patch embeddings into a single embedding. This will allow the model to learn global features.

### MLP Decoder

The model will push the patches through a series of transformer blocks, and take the output of all these blocks as input to the decoder. 

It will first push the multi-level features through an MLP to squeeze the channel dimension. Then, features are upsampled and concatenated. An MLP layer will then fuse the features together. Finally, another MLP layer will be used to predict the segmentation mask.