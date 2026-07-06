import torch
from mamba_ssm import Mamba


def main():
    print("Torch:", torch.__version__)
    print("CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    print("GPU:", torch.cuda.get_device_name(0))

    model = Mamba(
        d_model=64,
        d_state=16,
        d_conv=4,
        expand=2,
    ).cuda()

    x = torch.randn(2, 128, 64, device="cuda")
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Mamba test passed.")


if __name__ == "__main__":
    main()
