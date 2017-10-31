from collections import deque
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class DQNAgent:
    """docstring for DQNAgent"""
    def __init__(self, action_space, state_size):
        self.action_space = action_space
        self.state_size = state_size
        self.action_size = len(action_space)
        self.alpha = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1
        self.gamma = 0.995
        self.batch_size = 64
        self.train_start = 1000
        self.memory = deque(maxlen=2000)
        self.model = self.__build_model__()
        self.target_model = self.__build_model__()
        self.update_target_model()
    
    def __build_model__(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='softmax') )
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.alpha))
        return model

    def choose_an_action(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.model.predict(state)[0])

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def train_model(self):
        if len(self.memory) < self.train_start:
            return 

        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)
        update_input, update_target = [], []

        for state, action, reward, next_state, done in mini_batch:
            # print("{}, {}, {}, {}, {}".format(state, action, reward, next_state, done))
            target = self.model.predict(state)[0]
            next_state = np.reshape(next_state, [1, self.state_size])
            target_val = self.target_model.predict(next_state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(target_val)
            target = np.reshape(target, [1, self.action_size])
            update_target.append(target.tolist()[0])
            update_input.append(state.tolist()[0])
        self.model.fit(update_input, update_target, epochs=1, verbose=0)
            























