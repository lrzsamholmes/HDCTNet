# HDCTNet
The **H**ybrid **D**eformable **C**hannel **T**ransformer **N**etwork(HDCTNet) is designed to perform cell-level instance segmentation on mouse kidney tissue images. <br>
<br>
In this architecture, the deformable channel transformer block works as a bridge between the convolutional encoder and decoder. It is responsible for discovering cross channel feature dependencies in a more efficient way. <br>
<br>
![](/images/HDCTNet.png)
note: CNA refers to convolution, normalization and activation operation.<br>
<br>
## Convolutional encoder and decoder
The encoder and decoder have a typical 5-layer convolutional neural network architecture. The encoder consists of maxpool operation, convolution, normalization and activation to extract features, layer by layer. Meanwhile, the decoder uses  upsample operation, convolution, normalization and activation to restore the image resolution.<br>
<br>
![](/images/convolutional_encoder.png =30)<br>
The structure of convolutional encoder<br>
<br>
![](/images/convolutional_decoder.png)<br>
The structure of convolutional decoder<br>
<br>
