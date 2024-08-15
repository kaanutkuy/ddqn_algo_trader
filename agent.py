import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
import numpy as np
from replay_buffer import ReplayBuffer
from trading_env import TradingEnvironment


class Agent():
    def __init__(self, strategy, input_shape, num_actions=3, memory_size=7000, 
                 gamma=0.97, learning_rate=0.0001, dropout=0.3, batch_size=64, epsilon=1.0, 
                 epsilon_end=0.1, epsilon_decay=0.001, target_update_freq=100,
                 file_name="dqn_model.h5"):
        self.state_shape = input_shape
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        
        self.replay_buffer = ReplayBuffer(memory_size)
        
        self.strategy = strategy

        with self.strategy.scope():
          self.q_network = self.dqn_model(input_shape, num_actions, dropout)
          self.target_q_network = self.dqn_model(input_shape, num_actions, dropout)
        
          self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
          self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM)
          
          self.update_target_network()

        self.file_name = file_name

    @staticmethod
    def dqn_model(input_shape, num_actions=3, dropout=0.3) -> Sequential:
        """
        Shared Deep Q-Network architecture by the online and target agents.
        Used by the online agent to learn to give optimal trade actions which will be trained by the target agent that builds target Q-values.

        Args:
            input_shape (tuple(int, optional)): input state space shape
            num_actions (int, optional): number of possible actions to predict. Defaults to 3.

        Returns:
            Sequential: model to be trained and used by the agents
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(64, return_sequences=True),
            LSTM(32),
            Dropout(dropout),
            Dense(16, activation='elu'),
            Dense(num_actions)
        ])
        return model
        
    def update_target_network(self):
        self.target_q_network.set_weights(self.q_network.get_weights())
        
    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.num_actions)
        else:
            state = np.expand_dims(state, axis=0)
            q_values = self.q_network.predict(state)
            action = np.argmax(q_values)
        return action
    
    @tf.function
    def learn_step(self):
      if len(self.replay_buffer) < self.batch_size:
          return
        
      states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
      # Use online network to select actions
      q_next_values = self.q_network.predict(next_states)
      best_actions = np.argmax(q_next_values, axis=1)
          
      # Use target network to evaluate actions
      q_target_next_values = self.target_q_network.predict(next_states)
          
      # Compute target Q-values
      q_targets = rewards + self.gamma * q_target_next_values[np.arange(self.batch_size), best_actions] * (1 - dones)
          
      with tf.GradientTape() as tape:
        q_values = self.q_network(states)
        one_hot_actions = tf.one_hot(actions, self.num_actions)
        q_values = tf.reduce_sum(one_hot_actions * q_values, axis=1)
              
        loss = self.loss(q_targets, q_values)
          
      grads = tape.gradient(loss, self.q_network.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

      if self.epsilon > self.epsilon_min:
          self.epsilon -= self.epsilon_decay
      else:
          self.epsilon = self.epsilon_min

    def train(self, env: TradingEnvironment, num_episodes):
      best_reward = float("-inf")
        
      with self.strategy.scope():
        for episode in range(num_episodes):
          state = env.reset()
          episode_reward = 0
          done = False

          while not done:
            action = self.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            self.push_transition(state, action, reward, next_state, done)
            state = next_state
            self.strategy.run(self.learn_step)

          print(f'Episode: {episode}, Reward: {episode_reward}')
              
          if episode % self.target_update_freq == 0:
            self.update_target_network()
              
          # save best model
          if episode_reward > best_reward:
              best_reward = episode_reward
              self.save_model()

        print(f'Training completed. Best reward is: {best_reward}')

    def save_model(self):
        self.q_network.save(self.file_name)
    
    def load_model(self):
        self.q_network = load_model(self.file_name)
        
    def save_best_model(self, file_name="best_model.h5"):
        self.q_network.save(file_name)

    def load_best_model(self, file_name="best_model.h5"):
        self.q_network = load_model(file_name)
        self.target_q_network.set_weights(self.q_network.get_weights())
