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
	parser.add_argument('--logger-level', type=int, default=logging.DEBUG)
	parser.add_argument('--monitor', action='store_true')
	parser.add_argument('--update-interval', type=int, default=2048)
	parser.add_argument('--batchsize', type=int, default=64)
	parser.add_argument('--epochs', type=int, default=10)
	parser.add_argument('--entropy-coef', type=float, default=0.0) 
	parser.add_argument('--save_dir', type=str, default='nominal_inference',
						help='Directory to save inference results') 
	parser.add_argument('--goals_norm', type=float, default=1.0, 
						help='Minimum distance a goal should be generated from agent initial position')
	parser.add_argument('--load', type=str, default='nominal/PointGoal2.0-v1_0',
						help='Directory to load trained agent')
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
		env_seed = 2 ** 32 - 1 - args.seed if test else args.seed
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

	agent = PPO(model, opt,
		obs_normalizer=obs_normalizer, gpu=args.gpu,
		update_interval=args.update_interval, 
		minibatch_size=args.batchsize, epochs=args.epochs,
		clip_eps_vf=None, entropy_coef=args.entropy_coef,
		standardize_advantages=args.standardize_advantages,
		gamma=0.995,
		lambd=0.97,
		)

	print('Loading trained agent')
	agent.load(args.load)

	#function to sample new goals that are args.goal_norm away from agent initialization
	def gen_nonoverlap_goals(old_goal):
		env_obs = env.reset()
		env_goal = env.goal_pos
		while np.linalg.norm(old_goal - env_goal) < args.goals_norm:
			env_obs = env.reset()
			env_goal = env.goal_pos
		return env_obs

	#initialize list to log statistics and action magnitudes
	inference_stats = []
	total_action_mag = []
	for i in range(10):
		_ = env.reset()
		old_goal = env.goal_pos
		#generate new goals that is args.goal_norm away from agent
		env_obs = gen_nonoverlap_goals(old_goal)
		done = False
		t = 0.0
		env_R = 0.0
		env_r = 0.0
		nom_rate = 0
		action_mag = []
		while not done and t < 1000: #fails if more than 1000 for some environment reason
			action = agent.act(env_obs)
			env_obs, env_r, done, _ = env.step(action)
			env_R += env_r
			t += 1
			action_mag.append(action)
			# count number of times agent reaches goal
			if  env.dist_xy(env.goal_pos) <= 0.35:
				_ = env.reset()
				old_goal = env.goal_pos
				env_obs = gen_nonoverlap_goals(old_goal)
				nom_rate += 1

		print('Episode:{}, Nom_Reward:{}, Nom_Rate:{} '.format(i, env_R, nom_rate))
		total_action_mag.append(action_mag)
		inference_stats.append(np.asarray((i, env_R, nom_rate)))

	inference_stats = np.asarray(inference_stats)
	total_action_mag = np.asarray(total_action_mag)
	if os.path.exists(args.save_dir) == False:
		os.makedirs(args.save_dir)
	np.save(args.save_dir + '/' + args.env + '_' + str(args.seed) + '_'+ 'inference_stats.npy',inference_stats)
	np.save(args.save_dir + '/' + args.env + '_' + str(args.seed) + '_'+ 'action_mag.npy',total_action_mag)

if __name__ == '__main__':
	main()