# Repeat until convergence


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


def train(args) :



   env = gym.make(args.env_name)
   env.seed(args.seed)
   state_space = env.observation_space.shape[0]
   action_space = env.action_space.shape[0]

   print("state space :" , state_space , "  action_space :" ,  action_space)

   replay = ReplayBuffer(state_space, action_space)

   sac = SoftActorCritic(action_space, writer,learning_rate=args.learning_rate,alpha = args.alpha,
                      gamma=args.gamma, polyak=args.polyak)
   #sac.policy.load_weights(args.model_path + '/2020-05-30-19:03:13.833421/model')

   render = args.render
   start_steps=args.start_steps
   verbose=args.verbose
   epochs=args.epochs
   batch_size=args.batch_size

   global_step = 1
   episode = 1
   episode_rewards = []

   while True:

          # Observe state
          current_state = env.reset()

          step = 1
          episode_reward = 0
          done = False

          while not done:

              if render:
                  env.render()

              if global_step < start_steps:

                  action = env.action_space.sample()

              else:
                  action = sac.sample_action(current_state)

              # Execute action, observe next state and reward
              #print("actoin:" , action)
              next_state, reward, done, _ = env.step(action)#ros

              #print("fun",next_state.shape,current_state.shape)
              episode_reward +=  reward

              # Set end to 0 if the episode ends otherwise make it 1
              # although the meaning is opposite but it is just easier to mutiply
              # with reward for the last step.
              if done:
                  end = 0
              else:
                  end = 1

              if verbose:
                  logging.info(f'Global step: {global_step}')
                  logging.info(f'current_state: {current_state}')
                  logging.info(f'action: {action}')
                  logging.info(f'reward: {reward}')
                  logging.info(f'next_state: {next_state}')
                  logging.info(f'end: {end}')

              # Store transition in replay buffer
              replay.store(current_state, action, reward, next_state, end)

              # Update current state
              current_state = next_state

              step += 1
              global_step += 1



          if (step % 1 == 0) and (global_step > start_steps):
              for epoch in range(50):

                  # Randomly sample minibatch of transitions from replay buffer
                  current_states, actions, rewards, next_states, ends = replay.fetch_sample(num_samples=batch_size)

                  # Perform single step of gradient descent on Q and policy
                  # network
                  #print("in-----------------------------------------------------------",current_states.shape,next_states.shape)
                  critic1_loss, critic2_loss, actor_loss, alpha_loss = sac.train(current_states, actions, rewards, next_states, ends)
                  if verbose:
                      print(episode, global_step, epoch, critic1_loss.numpy(),
                            critic2_loss.numpy(), actor_loss.numpy(), episode_reward)


                  with writer.as_default():
                      tf.summary.scalar("actor_loss", actor_loss, sac.epoch_step)
                      tf.summary.scalar("critic1_loss", critic1_loss, sac.epoch_step)
                      tf.summary.scalar("critic2_loss", critic2_loss, sac.epoch_step)
                      tf.summary.scalar("alpha_loss", alpha_loss, sac.epoch_step)

                  sac.epoch_step += 1

                  if sac.epoch_step % 1 == 0:
                      sac.update_weights()


          if episode % 1 == 0:
              sac.policy.save_weights(args.model_path + f'{str(datetime.utcnow().date())}-{str(datetime.utcnow().time())}' + '/model')

          episode_rewards.append(episode_reward)
          episode += 1
          avg_episode_reward = sum(episode_rewards[-100:])/len(episode_rewards[-100:])

          print(f"Episode {episode} reward: {episode_reward}")
          print(f"{episode} Average episode reward: {avg_episode_reward}")
          with writer.as_default():
              tf.summary.scalar("episode_reward", episode_reward, episode)
              tf.summary.scalar("avg_episode_reward", avg_episode_reward, episode)


if __name__ == '__main__':
    args = argumant()

    train(args=args)
