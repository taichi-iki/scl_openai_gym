# coding: utf-8

import sys
import numpy as np
import gym

# python module for agent 
from a9 import Agent

# environment settings
MAX_TOTAL_T       = 200000
EPISODE_CUTOFF_T  = 50000
REWARD_SCALE      = 1.0
RENDERING_ENABLED = False
GAME_NAME         = 'Pong-v0'
# 'MontezumaRevengeNoFrameskip-v4'
# 'MontezumaRevenge-v0')
DEFAULT_LOG_NAME  = 'noname.log'

def reward_for_agent(r, is_trial_over):
    if is_trial_over:
        r -= REWARD_SCALE
    r = max(-1, min(REWARD_SCALE*r, 1))
    return r

if __name__ == '__main__':
    log_name = sys.argv[1] if len(sys.argv) >= 2 else DEFAULT_LOG_NAME
    
    agent = Agent()
    agent.start_subprocesses()
    
    env = gym.make(GAME_NAME)
    life_left = None
    ep = 0
    total_t = 0
    
    while True:
        ep_reward = 0
        # first observation
        obs = env.reset()
        for t in range(0, EPISODE_CUTOFF_T):
            if RENDERING_ENABLED: env.render()
            
            action = agent.next_step(obs)
            #action = env.action_space.sample()
            
            obs, reward_raw, done, info = env.step(action)
            ep_reward += reward_raw
            is_trial_over = (not life_left is None) and life_left > info['ale.lives']
            agent.reward(reward_for_agent(reward_raw, is_trial_over))
            
            if done:
                log_line = '%d\t%d\t%d\t%d\n'%(total_t, ep, t, ep_reward)
                with open(log_name, 'a') as f:
                    f.write(log_line)
                print('[ENV]', log_line)
                break
            
            # print('[ENV]', ep, t, total_t, reward_raw)
            total_t += 1
        
        if total_t >= MAX_TOTAL_T: break
        ep += 1
    
    agent.stop_subprocesses()
