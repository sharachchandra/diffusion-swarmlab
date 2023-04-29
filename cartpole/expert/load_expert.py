import os
import gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv 
import platform

### Select environment
env_id = "CartPole-v1"
# Note that the algorithm is SAC

env = make_vec_env(env_id, n_envs=1)
best_model = PPO.load('../../../rl-baselines3-zoo/rl-trained-agents/ppo/' + 
                      env_id + '_1/' + 
                      env_id + '.zip', env=env)
obs = env.reset()

gym_env = gym.make(env_id)
max_obs_values = gym_env.observation_space.high
min_obs_values = gym_env.observation_space.low

bias = max_obs_values + min_obs_values
bias = bias / 2
scale = max_obs_values - min_obs_values
scale = scale / 2

num_pairs = 1000000
expert_demo = np.zeros((num_pairs, obs.shape[1]+1))
i = 0
ep_return = 0
while i < num_pairs:
    if i%10000 == 0:
        print(i)
    action, _= best_model.predict(obs)
    sa_pair = np.concatenate((obs, [action-0.5]), axis=1).squeeze(0)
    # print(sa_pair)
    expert_demo[i] = sa_pair
    obs, reward, done, info = env.step(action)
    i += 1 
    ep_return += reward[0]   
    #print(obs, reward)
    #if done:
    #    print(ep_return)
    #    break 

expert_demo_path = 'logs/expert_demonstrations'

np.save(expert_demo_path, expert_demo)
expert_demo[:, :4] = expert_demo[:, :4] - bias
expert_demo[:, :4] = expert_demo[:, :4] / scale
#expert_demo[:, 3] = expert_demo[:, 3] - act_bias
#expert_demo[:, 3] = expert_demo[:, 3] / act_scale

normalized_expert_demo_path = 'logs/normalized_expert_demonstrations'
np.save(normalized_expert_demo_path, expert_demo)