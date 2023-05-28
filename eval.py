from datetime import datetime

import gym
from stable_baselines3 import DQN, PPO

from utils import CustomMLP

import my_own_env


side_length = 10
env = gym.make('Poll2D-v0', L=side_length, max_steps=20)

print("load model")
model_file = './results/pool2D_endreward_finish/2023-05-27T032305_L10/best/best_model.zip'
model = PPO.load(model_file)

# 
final_distances = []
rewards = []
for _ in range(4): 
    distances = []
    observation = env.reset()
    env.render() 
    for i in range(20):
        action, _ = model.predict(observation, deterministic=True)
        print(f"{env.step_no=}")
        observation, reward, done, info = env.step(action)
        print(f"{i=}, {action=}, {done=}, {reward=}, {env.distance_to_destiantion()=}")
        print(observation)
        distances.append(env.distance_to_destiantion())
        env.render()
        if done: 
            break
    print(distances)
    final_distances.append(distances[-1])
    rewards.append(reward)
    print("\n")
print(final_distances)
print(rewards)

model.policy