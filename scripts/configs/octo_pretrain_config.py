from copy import deepcopy
import imp
import os

from ml_collections import ConfigDict

get_base_config = imp.load_source(
    "config", os.path.join(os.path.dirname(__file__), "config.py")
).get_config

from octo.data.utils.text_processing import HFTokenizer
from octo.model.components.action_heads import DiffusionActionHead, L1ActionHead
from octo.model.components.tokenizers import (
    ImageTokenizer,
    LanguageTokenizer,
    LowdimObsTokenizer,
)
from octo.model.components.vit_encoders import SmallStem16
from octo.utils.spec import ModuleSpec
from octo.utils.train_utils import hf_weights_loader


def update_config(config, **kwargs):
    updates = ConfigDict(kwargs)
    new_config = deepcopy(config)
    new_config.update(updates)
    return new_config


def get_config(config_string=None):
    config = get_base_config(config_string)

    config["window_size"] = 1
    # config["num_steps"] = 300000
    config["num_steps"] = 50000
    config["shuffle_buffer_size"] = 1000
    config["model"]["observation_tokenizers"] = {
        "primary": ModuleSpec.create(
            ImageTokenizer,
            obs_stack_keys=["image_primary"],
            task_stack_keys=["image_primary"],
            encoder=ModuleSpec.create(SmallStem16),
        ),
        # "secondary": ModuleSpec.create(
        #     ImageTokenizer,
        #     obs_stack_keys=["image_secondary"],
        #     task_stack_keys=["image_secondary"],
        #     encoder=ModuleSpec.create(SmallStem16),
        # ),
        # "wrist": ModuleSpec.create(
        #     ImageTokenizer,
        #     obs_stack_keys=["image_wrist"],
        #     task_stack_keys=["image_wrist"],
        #     encoder=ModuleSpec.create(SmallStem16),
        # ),
        # "proprio": ModuleSpec.create(
        #     LowdimObsTokenizer,
        #     discretize=False,
        #     n_bins=256,
        #     obs_keys=["proprio"],
        # ),
    }
    config["model"]["task_tokenizers"] = {
        "language": ModuleSpec.create(
            LanguageTokenizer,
            encoder="t5-base",
            finetune_encoder=False,
        ),
    }
    config["model"]["repeat_task_tokens"] = True
    config["model"]["readouts"] = {"action": 1}
    config["model"]["heads"]["action"] = ModuleSpec.create(
        L1ActionHead,
        readout_key="readout_action",
        pred_horizon=10,
        action_dim=17,
        max_action=120.0,
    )

    # We augment differently for the primary and wrist cameras
    primary_augment_kwargs = dict(
        random_resized_crop=dict(scale=[0.8, 1.0], ratio=[0.9, 1.1]),
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_resized_crop",
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )
    wrist_augment_kwargs = dict(
        random_brightness=[0.1],
        random_contrast=[0.9, 1.1],
        random_saturation=[0.9, 1.1],
        random_hue=[0.05],
        augment_order=[
            "random_brightness",
            "random_contrast",
            "random_saturation",
            "random_hue",
        ],
    )

    # ML-collections complains if the type of an existing field changes
    # so we delete and re-add the field

    del config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"]
    del config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"]

    config["dataset_kwargs"]["frame_transform_kwargs"]["resize_size"] = {
        "primary": (256, 256),  # workspace camera is at 256x256
        # "secondary": (256, 256),
        # "wrist": (256, 256),
    }
    config["dataset_kwargs"]["frame_transform_kwargs"]["image_augment_kwargs"] = [
        primary_augment_kwargs,
        # have the same augmentations for the top camera
        # primary_augment_kwargs,
        # wrist_augment_kwargs,
    ]

    config["frame_transform_threads"] = 8

    config = update_config(
        config,
        optimizer=dict(
            frozen_keys=("*hf_model*",),
        ),
        dataset_kwargs=dict(
            oxe_kwargs=dict(
                data_mix="mimic",
                data_dir="/home/erbauer/tensorflow_datasets/",
                load_camera_views=(
                    "primary",
                    # "secondary",
                ),
                load_depth=False,
                load_proprio=False,
            ),
            traj_transform_kwargs=dict(
                future_action_window_size=9,
            ),
            batch_size=64,
            shuffle_buffer_size=1000,
            balance_weights=False,
        ),
        # {
        #     "name": "faive_dataset",
        #     "data_dir": "/home/erbauer/tensorflow_datasets/",
        #     "image_obs_keys": {"primary": "image", "top": "top_image"},
        #     "state_obs_keys": ["state"],
        #     "language_key": "language_instruction",
        #     "action_proprio_normalization_type": "normal",
        #     # All actions are relative deltas, except for the last one (gripper) which is absolute
        #     # Specifying this is only necessary if you want to predict > 1 step into the future
        #     # For Faive, 6 relative delta/twist outputs and 11 aboslute hand outputs
        #     "absolute_action_mask": [False] * 6 + [True] * 11,
        #     # standardize_fn is dynamically loaded from a file
        #     # for example: "experiments/kevin/custom_standardization_transforms.py:aloha_dataset_transform"
        #     # TODO: check if this is necessary for Faive
        #     # "standardize_fn": "octo/data/oxe/oxe_standardization_transforms.py:bridge_dataset_transform",
        #     # If the default data loading speed is too slow, try these:
        #     # "num_parallel_reads": 8,  # for reading from disk / GCS
        #     # "num_parallel_calls": 16,  # for initial dataset construction
        # },
        # dict(
        #     oxe_kwargs=dict(
        #         data_mix="oxe_magic_soup",
        #         data_dir="gs://rail-octo-central2/resize_256_256",
        #         load_camera_views=("primary", "wrist"),
        #         load_depth=False,
        #     ),
        #     traj_transform_kwargs=dict(
        #         future_action_window_size=3,
        #     ),
        #     batch_size=128,
        #     shuffle_buffer_size=500000,
        #     balance_weights=True,
        # ),
        text_processor=ModuleSpec.create(
            HFTokenizer,
            tokenizer_name="t5-base",
            encode_with_model=False,
            tokenizer_kwargs={
                "max_length": 16,
                "padding": "max_length",
                "truncation": True,
                "return_tensors": "np",
            },
        ),
        pretrained_loaders=(
            ModuleSpec.create(
                hf_weights_loader,
                hf_model="t5-base",
            ),
        ),
        eval_datasets=["faive_dataset"],
    )
    return config
