import os
import gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import platform

### Select environment
env_id = "MountainCarContinuous-v0"
# Note that the algorithm is SAC

gym_env = gym.make(env_id)
max_obs_values = gym_env.observation_space.high
min_obs_values = gym_env.observation_space.low

max_act_value = gym_env.action_space.high
min_act_value = gym_env.action_space.low

bias = max_obs_values + min_obs_values
bias = bias / 2
scale = max_obs_values - min_obs_values
scale = scale / 2

env = make_vec_env(env_id, n_envs=1)
best_model = SAC.load('../../../rl-baselines3-zoo/rl-trained-agents/sac/' + 
                      env_id + '_1/' + 
                      env_id + '.zip', env=env)
obs = env.reset()

num_pairs = 1000000
expert_demo = np.zeros((num_pairs, obs.shape[1]+1))
i = 0
ep_return = 0
num_steps = 0
while i < num_pairs:
    action, _= best_model.predict(obs)
    sa_pair = np.concatenate((obs, action), axis=1).squeeze(0)
    expert_demo[i] = sa_pair
    obs, reward, done, info = env.step(action)
    i += 1    
    ep_return += reward 
    num_steps += 1
    if done:
        print(ep_return)
        print(num_steps)
        ep_return = 0
        num_steps = 0
expert_demo_path = 'logs/expert_demonstrations'
np.save(expert_demo_path, expert_demo)
expert_demo[:, :2] = expert_demo[:, :2] - bias
expert_demo[:, :2] = expert_demo[:, :2] / scale
normalized_expert_demo_path = 'logs/normalized_expert_demonstrations'
np.save(normalized_expert_demo_path, expert_demo)
