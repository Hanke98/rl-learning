from myDqn import DQNAgent
from maze_env import Maze
import numpy as np


if __name__ == '__main__':
    action_space = [0, 1, 2, 3]
    dqn = DQNAgent(state_size=2, action_space=action_space)
    env = Maze()
    epochs = 0
    step = 0

    while epochs < 2000:
        state = env.reset()
        state = np.reshape(state, [1, len(state)])
        step = 0
        while True:
            env.render()
            action = dqn.choose_an_action(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, len(next_state)])
            dqn.append_sample(state, action, reward, next_state, done)
            dqn.train_model()
            state = next_state
            step += 1
            if done:
                dqn.update_target_model()
                break
        print("epochs: {}, step: {}, epsilon: {}".format(epochs, step, dqn.epsilon))
        epochs += 1
