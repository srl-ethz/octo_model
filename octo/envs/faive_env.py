import time

import gym
import numpy as np


def convert_obs(image_obs, qpos, im_size):
    """
    Image obs: (N, C, H, W)
    qpos: (17,)

    Convert observations to format usable by model.
    """
    transformed_image_obs = []
    for i, im in enumerate(image_obs):
        transformed_image_obs[i] = (
            im.reshape(3, im_size, im_size).transpose(1, 2, 0) * 255
        ).astype(np.uint8)

    # add padding to proprio to match training
    # proprio = np.concatenate([obs["state"][:6], [0], obs["state"][-1:]])
    proprio = qpos
    # NOTE: assume image_1 is not available
    return {
        "image_primary": transformed_image_obs[0],
        "image_top": transformed_image_obs[1],
        "image_wrist": transformed_image_obs[2],
        "proprio": proprio,
    }


def null_obs(img_size):
    return {
        "image_primary": np.zeros((img_size, img_size, 3), dtype=np.uint8),
        "image_top": np.zeros((img_size, img_size, 3), dtype=np.uint8),
        "image_wrist": np.zeros((img_size, img_size, 3), dtype=np.uint8),
        "proprio": np.zeros((8,), dtype=np.float64),
    }


class FaiveGym(gym.Env):
    """
    A Gym environment for the Faive hand.
    Needed to use Gym wrappers.
    """

    def __init__(
        self,
        policy_player_agent,
        im_size: int = 256,
    ):
        self.im_size = im_size
        self.observation_space = gym.spaces.Dict(
            {
                "image_primary": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "image_top": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "image_wrist": gym.spaces.Box(
                    low=np.zeros((im_size, im_size, 3)),
                    high=255 * np.ones((im_size, im_size, 3)),
                    dtype=np.uint8,
                ),
                "proprio": gym.spaces.Box(
                    low=np.ones((17,)) * -1, high=np.ones((17,)), dtype=np.float64
                ),
            }
        )
        self.action_space = gym.spaces.Box(
            low=np.ones((17,)) * -1, high=np.ones((17,)), dtype=np.float64
        )
        self.policy_player_agent = policy_player_agent

    def step(self, action):
        print(f'Action: {action}')
        # self.policy_player_agent.publish(
        #     hand_policy=action[6:], wrist_policy=action[:6]
        # )
        img_obs, qpos = self.policy_player_agent.get_current_observations()

        truncated = False

        obs = convert_obs(img_obs, qpos, self.im_size)

        return obs, 0, False, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.widowx_client.reset()

        self.is_gripper_closed = False
        self.num_consecutive_gripper_change_actions = 0

        image_obs, qpos = self.policy_player_agent.get_current_observations()
        obs = convert_obs(image_obs, qpos, self.im_size)

        return obs, {}
