import sys
sys.path.append('../')

import os
import gym
import numpy as np
 
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from diffusion_framework.ddpm_1d import DDPM_1d
from diffusion_framework.nets import CondErrorNet

device = "cuda" if torch.cuda.is_available() else "cpu"


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

# Define data
expert_demo_path = 'expert/logs/normalized_expert_demonstrations.npy'

dataset = np.load(expert_demo_path, allow_pickle=True)
dataset = torch.tensor(dataset, dtype=torch.float32)

required_dataset_size = int(sys.argv[1])
dataset = dataset[:required_dataset_size]

batch_size = 128
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define models
timesteps = 100
diffusion = DDPM_1d(timesteps)
model = CondErrorNet(dim=1,cond_dim=2)
model.to(device)

# training if the model is not available 
model_path = os.path.join('models', 'conditional', sys.argv[1] + '.pth')

if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))
else:
    optimizer = Adam(model.parameters(), lr=1e-4)
    epochs = 20

    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            batch_size = batch.shape[0]
            batch = batch.to(device)
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()
            loss = diffusion.conditional_p_losses(model, batch, t, loss_type="huber")
            if step % 100 == 0:
                print("Epoch: %d, Loss: %f" %(epoch, loss.item()))

            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), model_path)

num_test_episodes = 10 
avg_num_steps = 0 
success_rate = 0
avg_episode_return = 0
num_failure_steps = 200

for i in range(num_test_episodes):
    print('testing episode no : ', i)
    obs = gym_env.reset()
    done = False 
    num_steps = 0 
    episode_return = 0
    while not done:
        cond = torch.tensor((obs-bias)/scale).to(device)
        samples = diffusion.conditional_sample(cond.view(1,-1), model, image_size=1, batch_size=1)
        action = samples[-1][0]
        obs, reward, done, info = gym_env.step(action)
        num_steps += 1
        episode_return += reward

        if num_steps > num_failure_steps:
            done = True

    if episode_return > 0:
        success = 1
    else:
        success = 0   

    success_rate += success 
    avg_episode_return += episode_return
    avg_num_steps += num_steps

success_rate = success_rate / num_test_episodes
avg_episode_return = avg_episode_return / num_test_episodes
avg_num_steps = avg_num_steps / num_test_episodes

print("The success rate is %f " % success_rate)
print("The average episode return is  %f" % avg_episode_return) 
print("The average episode length is %d" % avg_num_steps) 