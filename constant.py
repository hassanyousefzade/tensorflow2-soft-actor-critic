
import argparse

def argumant() :

          parser = argparse.ArgumentParser(description='SAC')
          parser.add_argument('--seed', type=int, default=42,
                              help='random seed')
          parser.add_argument('--env_name', type=str, default='MountainCarContinuous-v0',
                              help='name of the gym environment with version')
          parser.add_argument('--render', type=bool, default=False,
                              help='set gym environment to render display')
          parser.add_argument('--verbose', type=bool, default=False,
                              help='log execution details')
          parser.add_argument('--batch_size', type=int, default=128,
                              help='minibatch sample size for training')
          parser.add_argument('--epochs', type=int, default=50,
                              help='number of epochs to run backprop in an episode')
          parser.add_argument('--start_steps', type=int, default=0,
                              help='number of global steps before random exploration ends')
          parser.add_argument('--model_path', type=str, default='../data/models/',
                              help='path to save model')
          parser.add_argument('--gamma', type=float, default=0.99,
                              help='discount factor for future rewards')
          parser.add_argument('--polyak', type=float, default=0.995,
                              help='coefficient for polyak averaging of Q network weights')
          parser.add_argument('--learning_rate', type=float, default=0.001,
                              help='learning rate')
          parser.add_argument('--alpha', type=float, default=0.2,
                              help='alpha')
                          
          return parser.parse_args()

