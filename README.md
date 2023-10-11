# HDCTNet
The **H**ybrid **D**eformable **C**hannel **T**ransformer **N**etwork(HDCTNet) is designed to perform cell-level instance segmentation on mouse kidney tissue images. <br>
<br>
In this architecture, the deformable channel transformer block works as a bridge between the convolutional encoder and decoder. It is responsible for discovering cross channel feature dependencies in a more efficient way. <br>
<br>
<div align=center><img src="/images/HDCTNet.png"></div><br>
note: CNA refers to convolution, normalization and activation operation.<br>
<br>
## Convolutional encoder and decoder
The encoder and decoder have a typical 5-layer convolutional neural network architecture. The encoder consists of maxpool operation, convolution, normalization and activation to extract features, layer by layer. Meanwhile, the decoder uses  upsample operation, convolution, normalization and activation to restore the image resolution.<br>
<br>
<div align=center><img width="200" src="/images/convolutional encoder.png"></div><br>
<p align=center>The structure of convolutional encoder</p><br>
<br>
<div align=center><img width="200" src="/images/convolutional decoder.png"></div><br>
<p align=center>The structure of convolutional decoder</p><br>
<br>
##Transformer block
The deformable channel-wise transformer is designed to fuse the cross-channel feature dependencies to alleviate the drawback of skip connection. The block consists of three parts: the embedding layer, the multi-head channel-wise attention block and the deformable network.<br>
<br>
