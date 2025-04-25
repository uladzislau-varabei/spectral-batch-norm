# Spectral Batch Normalization &mdash; unofficial PyTorch implementation 

Unofficial implementation of a normalization layer from paper
[Spectral Batch Normalization: Normalization in the Frequency Domain](https://arxiv.org/abs/2306.16999).

Some notes about this implementation:
- SBN layer and BN layer are combined into `SpatialSpectralBatchNorm2d` layer.
- Supported values for `fft_norm` are `[forward, backward, ortho, full]`. 
Mode `full` normalizes output  of `fft` and `ifft` and should be preferred to avoid possible issues 
with numerical stability.

Some notes from the paper: 
- Use SBN layer after BN layer.
- Use SBN layers in deeper blocks of model.
- ...
