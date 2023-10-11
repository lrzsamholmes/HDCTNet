# HDCTNet
The **H**ybrid **D**eformable **C**hannel **T**ransformer **N**etwork(HDCTNet) is designed to perform cell-level instance segmentation on mouse kidney tissue images. <br>
<br>
In this architecture, the deformable channel transformer block works as a bridge between the convolutional encoder and decoder. It is responsible for discovering cross channel feature dependencies in a more efficient way. <br>
<br>
<div align=center><img src="/images/HDCTNet.png"></div><br>
note: CNA refers to convolution, normalization and activation operation.<br>

## Convolutional encoder and decoder
The encoder and decoder have a typical 5-layer convolutional neural network architecture. The encoder consists of maxpool operation, convolution, normalization and activation to extract features, layer by layer. Meanwhile, the decoder uses  upsample operation, convolution, normalization and activation to restore the image resolution.<br>
<br>
<div align=center><img width="200" src="/images/convolutional encoder.png"></div><br>
<p align=center>The structure of convolutional encoder</p><br>
<br>
<div align=center><img width="200" src="/images/convolutional decoder.png"></div><br>
<p align=center>The structure of convolutional decoder</p><br>

## Transformer block
The deformable channel-wise transformer is designed to fuse the cross-channel feature dependencies to alleviate the drawback of skip connection. The block consists of three parts: the embedding layer, the multi-head channel-wise attention block and the deformable network.<br>
<br>
Before feeding the output of the encoder to the attention block,  the ferature maps need to be flattend to tokens, and transposed for channel-wise attention.<br>
<br>
<div align=center><img width="200" src="/images/embedding.png"></div><br>
<p align=center>The structure of embedding layer</p><br>
<br>
The structure of the deformable channel-wise attention is similar to the vanilla self-attention. However, the key and value matrices are no longer originated from one single layer input, but obtained from all levels of inputs. This modification helps to extract more inter-level connections.<br>
<br>
<div align=center><img width="200" src="/images/attention.png"></div><br>
<p align=center>The structure of deformable attention block</p><br>
<br>
Then the linear tokens are reconstructed to 2D feature maps by the reconstruction block.<br>
<br>
<div align=center><img width="200" src="/images/reconstruct.png"></div><br>
<p align=center>The structure of reconstruction layer</p><br>
<br>
The deformable attention block performs a convolution operation to the key and value matrices and calculates an offset map which indicates the focus point. The offset map is then added to the key and value matrices with bilinear interpolation, so that the attention operation can focus on the most valuable parts.<br>
<br>
<div align=center><img width="200" src="/images/channel-wise deformable.png"></div><br>
<p align=center>The structure of offset generation network</p><br>

## Example of dataset
<div align=center><img src="/images/dataset.png"></div><br>
<p align=center>On the left: mouse kidney tissue stained by periodic acid schiff(PAS)</p>
<p align=center>On the right: ground truth</p>
<p align=center>The colors indicate different biological structurtes.</p><br>

## Example of results
