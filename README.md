# DeepResNet_SR_MultipleEnhancement
Deep residual convolution network based super resolution algorithm which applies several mechanisms. 

### network architecture

The network can be split into three parts, base feature extraction, residual blocks and upsampler. The whole net has no global skip connections. We use a simple 3\*3 convolution layer for basic feature extracing, and employs a pixel-shuffle layer based upsampler which can realize arbitrary integral upscale factor. The residual block envolves following mechanism:

1. channel attention (refer to [Image Super-Resolution Using Very Deep Residual Channel Attention Networks](https://arxiv.org/abs/1807.02758v2), and we made a little modification.)
2. wide activation (refer to [Wide Activation for Efficient and Accurate Image Super-Resolution](<https://arxiv.org/abs/1808.08718>).)
3. wight normalization (also refer to [Wide Activation for Efficient and Accurate Image Super-Resolution](<https://arxiv.org/abs/1808.08718>).)
4. asymmetric decomposition of normal convolution layer. For example, a 3\*3 convolution has a receptive field of 3\*3 and has 9 parameters, a 5\*5 convolution has a receptive field of 5\*5 and has 25 parameters, but if we decompose 3\*3 convolution into a 5\*1 convolution and a 1*5 convolution, only 1 parameter is added, but 177% receptive field is increased. In some cases the LR images which input to the full convolution network may have large size, we believe enlarged receptive field can help in such situation.

### loss function

1. Inspired by [Loss Functions for Neural Networks for Image Processing](<https://arxiv.org/abs/1511.08861>), we tried several combinations of pixel-wise loss function, and found that the best combination changes according to the network size and not always l1 loss first and l2 after.
2. We explored a new loss function and named it as FFT loss, which is calculated on frequency domain: $l_{FFT}(I^{SR},I^{HR})=\Vert HighPass(FFT(I^{SR}))-HighPass(FFT(I^{HR})) \Vert_1$. Use this loss function to finetune the trained network can reduce flaw flawed details.

### training data

DIV2K or FLICKR2K dataset is both OK, but note that FLICKR2K is about 3 times the size of DIV2K. Crop the HR image into small patches and use matlab to generate corresponding LR patches. Generate a json file for training image pair list, format:

```txt
[
{'HR': '/SSD2/SR/data/train_fixHR_nooverlap/192/HR/DIV2K/0001_0.png', 'LR': '/SSD2/SR/data/train_fixHR_nooverlap/192/X3_BICUBIC/DIV2K/0001_0.png'},
{'HR': '/SSD2/SR/data/train_fixHR_nooverlap/192/HR/DIV2K/0001_1.png', 'LR': '/SSD2/SR/data/train_fixHR_nooverlap/192/X3_BICUBIC/DIV2K/0001_1.png'},
......
{'HR': '/SSD2/SR/data/train_fixHR_nooverlap/192/HR/DIV2K/0509_7.png', 'LR': '/SSD2/SR/data/train_fixHR_nooverlap/192/X3_BICUBIC/DIV2K/0509_7.png'},
......
{'HR': '/SSD2/SR/data/train_fixHR_nooverlap/192/HR/DIV2K/0800_9.png', 'LR': '/SSD2/SR/data/train_fixHR_nooverlap/192/X3_BICUBIC/DIV2K/0800_9.png'}
]
```

### validation data

Also generate a json file for every test set. Image cropping is not needed. While calculating PSNR and SSIM, although there is a little difference between python and matlab, we use python for convenience.

### train

When training data and validation data is prepared, training process can be simpliy launched like:

```bash
CUDA_VISIBLE_DEVICES=0,1 python3 trainer.py --upscale-factor=2 --optimizer=Adam --lr=1e-3 --lr-halve-step=60 --total-epochs=300 --batch-size=16 --loss=fftloss --checkpoint-dir=checkpoints --save-subdir=test --save-prefix=1 | tee 2>&1 log.txt
```

