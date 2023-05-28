from datetime import datetime

import gym

from gym.wrappers import TimeLimit
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.env_util import make_vec_env

from utils import CustomMLP

import my_own_env


side_length = 10

date = datetime.now().strftime("%Y-%m-%dT%H%M%S")
folder_path = f"./results/pool2D_endreward_finish/{date}_L{side_length}"

env = gym.make('Poll2D-v0', L=side_length, max_steps=20)
env = TimeLimit(env, max_episode_steps=100)
check_env(env)
vec_env = make_vec_env('Poll2D-v0', n_envs=8, env_kwargs=dict(L=side_length, max_steps=20))

policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=32, side_length=side_length)
)

model = PPO(
    "MlpPolicy",
    vec_env,
    policy_kwargs=policy_kwargs,
    verbose=True,
    learning_rate=0.0001, 
    batch_size=128,
#    n_steps=4096,
#    n_epochs=3,
)
print(model.policy.features_extractor.mlp)

eval_callback = EvalCallback(
    env,
    best_model_save_path=folder_path + "/best/",
    log_path=folder_path + "/logs/",
    n_eval_episodes=20,
    deterministic=False,
    render=False,
)

new_logger = configure(folder_path + "/logs/", ["stdout", "csv"])  # "tensorboard"
model.set_logger(new_logger)


print("start learning")
model.learn(total_timesteps=1000_000, callback=eval_callback)
model.save(f"{folder_path}/model")

# 
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
    print("\n")

model.policy
