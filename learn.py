from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from gym.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

import my_own_env

side_length = 9
env = gym.make('Poll2D-v0', L=side_length, max_steps=20)
env = TimeLimit(env, max_episode_steps=100)

check_env(env)


class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        mlp_hidden_dim = 16
        #self.input_dim = int(observation_space.shape[0])
        self.input_dim = int(observation_space.shape[0]*side_length)
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
        )
        self.linear = nn.Sequential(nn.Linear(mlp_hidden_dim, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # observations are one-hot encoded
        #obs = torch.argmax(torch.reshape(observations, (self.input_dim,-1)), dim=1)/(side_length-1)  # one hot to float: argmax to decode
        #r = self.linear(self.mlp(obs))
        #return torch.reshape(r,(1,-1))
        r = self.linear(self.mlp(observations))
        return r

policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=8),
)

date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
folder_path = f"./results/pool2D_endreward_finish/{date}_L{side_length}"
# model = PPO("MlpPolicy", env, tensorboard_log=folder_path)
model = PPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=True,
    learning_rate=0.0003, 
    batch_size=128
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=folder_path + "/best/",
    log_path=folder_path + "/logs/",
    n_eval_episodes=20,
    deterministic=True,
    render=False,
)

print("start learning")

print(model.policy.features_extractor.mlp)
model.learn(total_timesteps=30_000, callback=eval_callback)
model.save(f"{folder_path}/model")


distances = []
observation = env.reset()
env.render()
for i in range(10):
    action, _ = model.predict(observation, deterministic=True)
    print(f"{env.step_no=}")
    observation, reward, done, info = env.step(action)
    print(f"{i=}, {action=}, {done=}, {reward=}, {env.distance_to_destiantion()=}")
    print(observation)
    distances.append(np.sqrt((env.destination[0]-env.position[0])**2+(env.destination[1]-env.position[1])**2))
    env.render()
    if done: 
        break
    
print(distances)

model.policy
