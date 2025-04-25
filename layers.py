import torch
import torch.nn as nn


class SpectralBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, fft_norm="backward", affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum  # default momentum the same as for batch norm layer
        assert fft_norm in ["forward", "backward", "ortho"]
        self.fft_norm = fft_norm
        # Shape of running mean/variance and affine params
        shape = (1, self.num_features, 1, 1)
        # Mean must be complex,so for better serialization split it into real/im parts
        self.register_buffer("mean_running_real", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("mean_running_im", torch.zeros(shape, dtype=torch.float32))
        self.register_buffer("var_running", torch.ones(shape, dtype=torch.float32))
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.ones(shape, dtype=torch.float32))
            self.bias = nn.Parameter(torch.zeros(shape, dtype=torch.float32))
        else:
            self.weight = None
            self.bias = None

    def update_running_stats(self, mean, var, x_fft):
        # Running mean must be stored as real/im parts
        mean_running = torch.complex(self.mean_running_real, self.mean_running_im)
        mean_running = self.momentum * mean + (1 - self.momentum) * mean_running
        # Update running mean real/im parts
        self.mean_running_real = mean_running.real
        self.mean_running_im = mean_running.imag
        # Variance must be unbiased, N = C * H * W / 2
        var_N = x_fft[0].numel()
        var_unbiased = var * (var_N / (var_N - 1))
        self.var_running = self.momentum * var_unbiased + (1 - self.momentum) * self.var_running

    def forward(self, x):
        # Input: N, C, H, W
        assert x.ndim == 4
        N, C, H, W = x.size()
        orig_dtype = x.dtype
        # It's better to run normalization in default precision
        x = x.to(dtype=torch.float32)
        # 1. Compute FFT 2d
        x_fft = torch.fft.rfft2(x, s=(H, W), dim=(-2, -1), norm=self.fft_norm)
        # 2. Compute mean/variance over minibatch across channels (only in training mode)
        if self.training:
            # Mean is computed separately for real and imaginary parts
            mean_real = torch.mean(x_fft.real, dim=(0, 2, 3), keepdim=True)
            mean_im = torch.mean(x_fft.imag, dim=(0, 2, 3), keepdim=True)
            mean = torch.complex(mean_real, mean_im)  # [1, C, 1, 1]
            var = torch.mean((x_fft - mean).abs().pow(2), dim=(0, 2, 3), keepdim=True)  # [1, C, 1, 1]
        else:
            mean = torch.complex(self.mean_running_real, self.mean_running_im)
            var = self.var_running
        # 3. Compute running mean/variance (only in training mode)
        if self.training:
            self.update_running_stats(mean, var, x_fft)
        # 4. Normalize
        x_fft = (x_fft - mean) / torch.sqrt(var + self.eps)
        # 5. Apply affine transform across channels
        if self.affine:
            x_fft = x_fft * self.weight + self.bias
        # 6. Compute inverse FFT 2d
        # It's better to provide original signal size as stated in the documentation
        x = torch.fft.irfft2(x_fft, s=(H, W), dim=(-2, -1), norm=self.fft_norm)
        x = x.to(dtype=orig_dtype)
        return x
