import tensorflow as tf
from tensorflow import keras
import numpy as np
import random
from collections import deque


class ReplayBuffer():
    def __init__(self, memory_capacity:int):
        self.buffer = deque(maxlen=memory_capacity)

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        experiences = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        
        states = np.array([np.array(s) for s in states])  # Handle potential nested lists/tuples
        actions = np.array(actions, dtype=np.int32)  # Specify integer type for actions
        rewards = np.array(rewards, dtype=np.float32)  # Specify float type for rewards
        next_states = np.array([np.array(s) for s in next_states])  # Handle potential nested lists/tuples
        dones = np.array(dones, dtype=np.bool_)  # Specify boolean type for dones

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
