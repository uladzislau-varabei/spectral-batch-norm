import numpy as np
import torch

from layers import SpectralBatchNorm2d, SpatialSpectralBatchNorm2d


def compute_base_stats(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = x.flatten()
    x_mean = x.mean()
    x_std = x.std()
    x_med = np.median(x)
    q = 0.1
    x_q1 = np.quantile(x, q)
    x_q2 = np.quantile(x, 1 - q)
    x_min = x.min()
    x_max = x.max()
    stats_message = f"mean={x_mean:.3f}, std={x_std:.3f}, med={x_med:.3f}, " \
                    f"min={x_min:.3f}, max={x_max:.3f}, q1={x_q1:.3f}, q2={x_q2:.3f}"
    return stats_message


def test_sbn_layer_forward(x_input, device, train_mode):
    print(f"--- SpectralBatchNorm2d (train_mode={train_mode}) ---")
    kwargs = dict(fft_norm="backward", affine=True)
    layer = SpectralBatchNorm2d(num_features=x_input.shape[1], **kwargs).to(device)
    if train_mode:
        layer.train()
    else:
        layer.eval()
    x_output = layer(x_input)
    shape_input = tuple(x_input.shape)
    shape_output = tuple(x_output.shape)
    assert shape_input == shape_output, f"Different shapes: x_input.shape={shape_input}, x_output.shape={shape_output}"
    print(f"Layer kwargs: {kwargs}")
    print(f"\nInput stats:\n{compute_base_stats(x_input)}")
    print(f"\nOutput stats:\n{compute_base_stats(x_output)}\n")


def test_ssbn_layer_forward(x_input, device, train_mode):
    print(f"--- SpectralSpatialBatchNorm2d (train_mode={train_mode}) ---")
    kwargs = dict(fft_norm="backward", affine=True)
    layer = SpatialSpectralBatchNorm2d(num_features=x_input.shape[1]).to(device)
    if train_mode:
        layer.train()
    else:
        layer.eval()
    x_output = layer(x_input)
    shape_input = tuple(x_input.shape)
    shape_output = tuple(x_output.shape)
    assert shape_input == shape_output, f"Different shapes: x_input.shape={shape_input}, x_output.shape={shape_output}"
    print(f"Layer kwargs: {kwargs}")
    print(f"\nInput stats:\n{compute_base_stats(x_input)}")
    print(f"\nOutput stats:\n{compute_base_stats(x_output)}\n")


if __name__ == "__main__":
    print("Running tests...")
    N, C, H, W = 4, 128, 64, 64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_input = torch.randn((N, C, H, W), dtype=torch.float32).to(device)
    test_sbn_layer_forward(x_input, device, train_mode=True)
    test_sbn_layer_forward(x_input, device, train_mode=False)
    test_ssbn_layer_forward(x_input, device, train_mode=True)
    test_ssbn_layer_forward(x_input, device, train_mode=False)
    print("--- All tests passed! ---")
