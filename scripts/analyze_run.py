"""
This script is for analyzing data that was created using the FaiveGym environment in /.../octo/envs/faive_env.py

This is a preliminary viz solution to debug model runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os

wrist_axis_names = [
    "delta rot x",
    "delta rot y",
    "delta rot z",
    "delta trans x",
    "delta trans y",
    "delta trans z",
]
gripper_axis_names = [f"gripper.angle.{i}" for i in range(11)]
axis_names = wrist_axis_names + gripper_axis_names


def find_files(root_dir):
    numpy_file = glob(os.path.join(root_dir, "*.npy"))[0]
    video_file = glob(os.path.join(root_dir, "*.mp4"))[0]

    # there should be a dict in there
    data = np.load(numpy_file, allow_pickle=True).item()
    return data


def plot_axes(data, axes, save_dir, wrist_cmd_type="delta"):
    pred_actions = np.array(data["pred_actions"])
    gt_actions = np.array(data["gt_actions"])

    print(f"Pred actions shape: {pred_actions.shape}")
    print(f"GT actions shape: {gt_actions.shape}")

    # Plot each axis separately
    for axis_id in axes:
        plt.figure(figsize=(8, 6))
        plt.plot(pred_actions[:, axis_id], label="Predicted", color="blue")
        plt.plot(gt_actions[:, axis_id], label="Ground Truth", color="red")
        plt.title(f"Axis: {axis_names[axis_id]}")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        plt.savefig(os.path.join(save_dir, f"axis_{axis_id}_plot.png"))
        plt.close()

    if wrist_cmd_type == "delta":
        pred_proprio = np.array(data["pred_next_proprio"])
        gt_proprio = np.array(data["gt_next_proprio"])

        # Plot each axis separately
        for axis_id in axes:
            print(f"Pred proprio shape: {pred_proprio.shape}")
            pred_proprio = pred_proprio.squeeze()
            plt.figure(figsize=(8, 6))
            plt.plot(pred_proprio[:, axis_id], label="Predicted", color="blue")
            plt.plot(gt_proprio[:, axis_id], label="Ground Truth", color="red")
            plt.title(f"Delta prediction: {axis_names[axis_id]}")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.grid(True)
            plt.tight_layout()

            # Save the plot
            plt.savefig(os.path.join(save_dir, f"delta_{axis_id}_plot.png"))
            plt.close()


if __name__ == "__main__":
    root_dir = "/home/erik/msthesis/saved_videos/"
    save_dir = root_dir
    axes = list(range(17))

    data = find_files(root_dir)
    plot_axes(data, axes, save_dir)
