import torch
import numpy as np
import pickle
import cv2
import os

from rlkit.utils.load_env import load_metagym_env, load_gym_env
from rlkit.policy import PPOPolicy
from typing import Optional, Dict, List

def save_rendering(path, e, recorded_frames):
    directory = os.path.join(path, 'video')
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_name = 'rendering' + str(e) +'.avi'
    output_file = os.path.join(directory, file_name)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI file
    fps = 120
    width = 480
    height = 480
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    for frame in recorded_frames:
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()
    recorded_frames = []

env_type = 'MetaGym'
agent_type = 'MT1'

key = '-'.join((env_type, agent_type))
task = 'pick-place'

if agent_type == 'MetaGym':
    _, env, _ = load_metagym_env(key)    
elif agent_type == 'Gym':
    _, env, _ = load_gym_env(key, task)    

# import pre-trained model before defining actual models
try:
    actor, critic, encoder, running_state = pickle.load(open('model/model.p', "rb"))
except:
    actor, critic, encoder = pickle.load(open('model/model.p', "rb"))

masking_indices = [None]

policy = PPOPolicy(
            actor=actor,
            critic=critic,
            encoder=encoder,
            optimizer=None,
            K_epochs=None,
            eps_clip=None,
            device=torch.device('cpu')
        )

policy.eval()
num_episodes = 0
seed = 0

for _ in range(1):
    try:
        s, _ = env.reset(seed=seed)
    except:
        s = env.reset(seed=seed)

    eval_ep_info_buffer = []
    recorded_frames = []
    episode_reward, episode_cost, episode_length, episode_success = 0, 0, 0, 0

    done = False
    while not done:
        with torch.no_grad():
            a, _ = policy.select_action(np.delete(s, masking_indices), deterministic=True)
        try:
            ns, rew, trunc, term, infos = env.step(a.flatten()); cost = 0.0
            done = term or trunc
        except:
            ns, rew, term, infos = env.step(a.flatten()); cost = 0.0
            done = term
        success = infos['success']
        
        mask = 0 if done else 1
        
        recorded_frames.append(env.render())
        
        episode_reward += rew
        episode_cost += cost
        episode_success += success
        episode_length += 1
        
        s = ns

        if done:
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward, "episode_cost": episode_cost, "episode_length": episode_length, "episode_success_rate":episode_success/episode_length}
            )
            num_episodes +=1
            episode_reward, episode_cost, episode_length = 0, 0, 0

        save_rendering('model/video')

result = {
    "eval/episode_reward": [ep_info["episode_reward"] for ep_info in eval_ep_info_buffer],
    "eval/episode_cost": [ep_info["episode_cost"] for ep_info in eval_ep_info_buffer],
    "eval/episode_length": [ep_info["episode_length"] for ep_info in eval_ep_info_buffer],
    "eval/episode_success_rate": [ep_info["episode_success_rate"] for ep_info in eval_ep_info_buffer],
}

#print(f'reward mean/std: {np.mean(result['eval/episode_reward'])}/{np.std(result['eval/episode_reward'])}')
#print(f'cost mean/std: {np.mean(result['eval/episode_cost'])}/{np.std(result['eval/episode_cost'])}')
#print(f'length mean/std: {np.mean(result['eval/episode_length'])}/{np.std(result['eval/episode_length'])}')
#print(f'success mean/std: {np.mean(result['eval/episode_success_rate'])}/{np.std(result['eval/episode_success_rate'])}')







