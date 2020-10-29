from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from builtins import *  # NOQA
from future import standard_library
standard_library.install_aliases()  # NOQA
import argparse
import os

import gym
import gym.wrappers
import numpy as np
from functools import partial

import chainer
from chainer import functions as F
from chainer import links as L
from chainer import Sequential
import chainerrl
from chainerrl.agents import PPO
from chainerrl import links, distribution, policies, experiments, misc
from chainerrl.optimizers.nonbias_weight_decay import NonbiasWeightDecay
from chainerrl.initializers import LeCunNormal
from chainerrl.policy import Policy

import safety_gym

def main():
	import logging

	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--env', type=str, default= 'PointGoal2.0-v1', choices=('PointGoal2.0-v1', 'CarGoal2.0-v1')) 
	parser.add_argument('--bound-mean', type=bool, default=True)
	parser.add_argument('--seed', type=int, default=0,
						help='Random seed [0, 2 ** 32)')
	parser.add_argument('--steps', type=int, default=5000000)
	parser.add_argument('--reward-scale-factor', type=float, default=1e-2)
	parser.add_argument('--standardize-advantages', action='store_true')
	parser.add_argument('--render', default=False)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--weight-decay', type=float, default=0)
	parser.add_argument('--load', type=str, default='')
	parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
	parser.add_argument('--monitor', action='store_true')
	parser.add_argument('--update-interval', type=int, default=2048)
	parser.add_argument('--batchsize', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--entropy-coef', type=float, default=0.0) 
	parser.add_argument('--save_dir', type=str, default='nominal') 
	args = parser.parse_args()

	logging.basicConfig(level=args.logger_level)
	misc.set_random_seed(args.seed, gpus=(args.gpu,))

	#duplicate safety-gym configs but extend gym dimensions and simplify lidar to cover entire space
	if args.env=='PointGoal2.0-v1':
		env = gym.make('Safexp-PointGoal0-v0')
	elif args.env=='CarGoal2.0-v1':
		env = gym.make('Safexp-CarGoal0-v0')
	config = env.config
	config['placements_extents']= [-2.0, -2.0, 2.0, 2.0]
	config['lidar_max_dist'] = 8*config['placements_extents'][3]
	from safety_gym.envs.engine import Engine
	env = Engine(config)
	from gym.envs.registration import register
	register(id=args.env,
	         entry_point='safety_gym.envs.mujoco:Engine',
	         kwargs={'config': config})

	def make_env(args_env,test):
		env = gym.make(args.env)
		# Use different random seeds for train and test envs
		env_seed = args.seed + 1 if test else args.seed
		env.seed(env_seed)
		# Cast observations to float32 because our model uses float32
		env = chainerrl.wrappers.CastObservationToFloat32(env)
		if args.monitor:
			env = gym.wrappers.Monitor(env, args.outdir)
		if not test:
			# Scale rewards (and thus returns) to a reasonable range so that training is easier
			env = chainerrl.wrappers.ScaleReward(env, args.reward_scale_factor)
		return env

	env = make_env(args.env, test=False)
	obs_space = env.observation_space
	action_space = env.action_space
	action_size = action_space.low.size

	# Normalize observations based on their empirical mean and variance
	obs_normalizer = chainerrl.links.EmpiricalNormalization(obs_space.low.size, clip_threshold=5)

	winit = chainerrl.initializers.Orthogonal(1.)
	winit_last = chainerrl.initializers.Orthogonal(1e-2)
	policy = chainer.Sequential(
				L.Linear(None, 64, initialW=winit), 
				F.tanh,
				L.Linear(None, 64, initialW=winit),
				F.tanh,
				L.Linear(None, action_size, initialW=winit_last),
				chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
					action_size=action_size,
					var_type='diagonal',
					var_func=lambda x: F.exp(2 * x),  # Parameterize log std
					var_param_init=0,  # log std = 0 => std = 1
					),
				)

	vf = chainer.Sequential(
				L.Linear(None, 128, initialW=winit),
				F.tanh,
				L.Linear(None, 64, initialW=winit),
				F.tanh,
				L.Linear(None, 1, initialW=winit),
				)
	
	model = chainerrl.links.Branched(policy, vf)

	if args.gpu > -1:
		import cupy as cp
		model.to_gpu(args.gpu)

	opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
	opt.setup(model)

	if args.weight_decay > 0:
		opt.add_hook(NonbiasWeightDecay(args.weight_decay))

	agent = PPO(model, opt,
		obs_normalizer=obs_normalizer, gpu=args.gpu,
		update_interval=args.update_interval, 
		minibatch_size=args.batchsize, epochs=args.epochs,
		clip_eps_vf=None, entropy_coef=args.entropy_coef,
		standardize_advantages=args.standardize_advantages,
		gamma=0.995,
		lambd=0.97)

	#stepwise training
	env_Rs = []
	i = 0
	while i < args.steps:
		env_obs = env.reset()
		done = False
		env_R = 0
		env_r = 0
		while not done:
			action = agent.act_and_train(env_obs, env_r)
			env_obs, env_r, done, info = env.step(action)
			env_R += env_r
			i+=1
		env_Rs.append(env_R)
		print('statistics:', agent.get_statistics())
		print('Rewards for current episode:', env_R)
		print('Steps:',i)
		agent.stop_episode_and_train(env_obs, env_r, done)

	stats = np.array(env_Rs, dtype=float)
	if os.path.exists(args.save_dir) == False:
		os.makedirs(args.save_dir)
	np.save(args.save_dir + '/' + args.env + '_'+ str(args.seed) + '.npy', stats)
	agent.save(args.save_dir + '/' + args.env + '_'+ str(args.seed))

if __name__ == '__main__':
	main()