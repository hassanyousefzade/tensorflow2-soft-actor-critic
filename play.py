import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import Model
import numpy as np
import gym
import argparse
import logging
from datetime import datetime
from constant import argumant
from replay_buffer import ReplayBuffer
from Model import SoftActorCritic
writer = tf.summary.create_file_writer('../data/models/' + f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}' + '/summary')

tf.keras.backend.set_floatx('float64')

def play(args) :

 while True:



    # Instantiate the environment.
    env = gym.make(args.env_name)
    env.seed(args.seed)
    state_space = env.observation_space.shape[0]
    # TODO: fix this when env.action_space is not `Box`
    action_space = env.action_space.shape[0]

    sac = SoftActorCritic(action_space, writer,learning_rate=args.learning_rate,alpha = args.alpha,
                      gamma=args.gamma, polyak=args.polyak)
                      
    sac.policy.load_weights(args.model_path + f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}' + '/model')



    # Observe state
    current_state = env.reset()

    episode_reward = 0
    done = False
    while not done:

        if args.render:
            env.render()

        action = sac.sample_action(current_state)

        # Execute action, observe next state and reward
        next_state, reward, done, _ = env.step(action)

        episode_reward +=  reward

        # Update current state
        current_state = next_state
if __name__ == '__main__':
    args = argumant()

    play(args=args)
