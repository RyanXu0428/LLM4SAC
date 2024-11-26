import pandas as pd
import torch
import rospy
from task_env import VrxAutoDockingEnv
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import cv2
from transformers import AutoImageProcessor, AutoModelForImageClassification
import ollama
from langchain.chains import ConversationChain
from sentence_transformers import SentenceTransformer, util
import pandas
import json
import time
from utils import get_prompt
import gymnasium as gym
import gymnasium.spaces as spaces
from custom_sac import SAC
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torchvision.models as models
from sentence_transformers import SentenceTransformer, util

class CustomPPO(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomPPO, self).__init__(observation_space, features_dim)

        self.fe = models.mobilenet_v2(pretrained=True)
        number_feature = self.fe.classifier[-1].in_features
        self.fe.classifier[-1] = nn.Linear(number_feature, 2)
        self.fe.load_state_dict(th.load("models/fe.pt"))
        #
        # for param in self.fe.parameters():
        #     param.requires_grad = False

    def forward(self, observations: th.Tensor) -> th.Tensor:

        img_data = observations[:, 21:]
        sensor_data = observations[:, :21]
        batch = observations.shape[0]
        img_data = img_data.reshape((batch, 3, 64, 64))

        img_tensor = self.fe(img_data)
        out = th.cat((img_tensor, sensor_data), dim=1)
        return out

if __name__ == "__main__":
    rospy.init_node('llmppo_trainer_node', anonymous=True, log_level=rospy.INFO)

    # 初始化环境
    env = gym.make('auto_docking_v0', apply_api_compatibility=True)

    policy_kwargs_ppo = dict(
        features_extractor_class=CustomPPO,
        features_extractor_kwargs=dict(features_dim=21),
        net_arch=[64, 64],
        activation_fn=th.nn.ReLU
    )

    model = SAC("MlpPolicy", env, policy_kwargs=policy_kwargs_ppo,
                verbose=1, learning_rate=0.0005, tensorboard_log="log",
                device="auto", batch_size=64, gamma=0.99)

    model.learn(150000)

    model.save("llm_sac.zip")



