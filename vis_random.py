import argparse
import einops
import math

import matplotlib.pyplot as plt
import torch

from ml.processing import perturb_vector, random_rotation

def sample(base_axis: torch.Tensor, vector_to_rotate: torch.Tensor, n_samples: int, angle_rad_std: float) -> torch.Tensor:
    # base_normal = torch.tensor([0.0, 0.0, 1.0])
    # samples = [perturb_vector(base_normal, normal_std) for _ in range(n_samples)]
    vector_to_rotate = einops.rearrange(vector_to_rotate, "c -> c 1")

    samples = []
    for _ in range(n_samples):
        x = random_rotation(vector_to_rotate, base_axis, angle_rad_std=angle_rad_std)
        samples.append(x)

    return torch.stack(samples)


def plot(base_axis: torch.Tensor, vector_to_rotate: torch.Tensor, rotated: torch.Tensor) -> None:
    xyz = rotated.numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Arrows from origin to each sampled direction
    zeros = torch.zeros_like(rotated)
    ax.quiver(
        zeros[:, 0].numpy(),
        zeros[:, 1].numpy(),
        zeros[:, 2].numpy(),
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        length=1.0,
        normalize=True,
        color="blue",
        linewidth=0.8,
        alpha=0.8,
    )

    zeros = torch.zeros(1, 3)
    ax.quiver(
        zeros[:, 0].numpy(),
        zeros[:, 1].numpy(),
        zeros[:, 2].numpy(),
        vector_to_rotate[None, 0],
        vector_to_rotate[None, 1],
        vector_to_rotate[None, 2],
        length=1.0,
        normalize=True,
        color="green",
        linewidth=2.0,
        alpha=0.8,
    )

    zeros = torch.zeros(1, 3)
    ax.quiver(
        zeros[:, 0].numpy(),
        zeros[:, 1].numpy(),
        zeros[:, 2].numpy(),
        base_axis[None, 0],
        base_axis[None, 1],
        base_axis[None, 2],
        length=1.0,
        normalize=True,
        color="red",
        linewidth=2.0,
        alpha=0.8,
    )

    # Faint unit sphere for reference
    u = torch.linspace(0, 2 * math.pi, 30)
    v = torch.linspace(0, math.pi, 15)
    x = torch.outer(torch.cos(u), torch.sin(v))
    y = torch.outer(torch.sin(u), torch.sin(v))
    z = torch.outer(torch.ones_like(u), torch.cos(v))
    ax.plot_surface(x, y, z, color="lightgray", alpha=0.15, linewidth=0, zorder=0)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Random normals (processing.random_rotation)")
    plt.tight_layout()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize random normals used by random_rotation()")
    parser.add_argument("-n", "--num", type=int, default=500, help="number of samples")
    parser.add_argument("--std", type=float, default=0.35, help="normal_std used for perturbation")
    args = parser.parse_args()

    base_axis = torch.tensor([0.0, 0.0, 1.0])
    vector_to_rotate = torch.tensor([1.0, 0.0, 0.0])
    rotated = sample(base_axis, vector_to_rotate, args.num, args.std)
    plot(base_axis, vector_to_rotate, rotated)


if __name__ == "__main__":
    main()
