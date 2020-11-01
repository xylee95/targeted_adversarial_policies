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
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--env', type=str, default= 'PointGoal2.0-v1')
	parser.add_argument('--bound-mean', type=bool, default=True)
	parser.add_argument('--seed', type=int, default=0,
						help='Random seed [0, 2 ** 32)')
	parser.add_argument('--outdir', type=str, default='results',
						help='Directory path to save output files.'
							 ' If it does not exist, it will be created.')
	parser.add_argument('--reward-scale-factor', type=float, default=1e-2) #default is 1e-2
	parser.add_argument('--standardize-advantages', action='store_true')
	parser.add_argument('--render', type=bool, default=False)
	parser.add_argument('--lr', type=float, default=3e-4)
	parser.add_argument('--load', type=str, default='robust/weights/PointGoal2.0-v1_3_0')
	parser.add_argument('--adv_load', type=str, default='adversary/weights/PointGoal2.0-v1_A_0')
	parser.add_argument('--monitor', action='store_true')
	parser.add_argument('--update-interval', type=int, default=2048)
	parser.add_argument('--batchsize', type=int, default=64)
	parser.add_argument('--goals_norm', type=float, default=1.0)
	parser.add_argument('--variant', type=int, default=3, choices=(1,2,3),
						help='ID to log inference results of robust policy trained with different variants of adv. training') 
	parser.add_argument('--save_dir', type=str, default='robust_inference',
						help='Directory to save inference results')

	args = parser.parse_args()

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
	obs_normalizer = chainerrl.links.EmpiricalNormalization(
		obs_space.low.size, clip_threshold=5)

	# action | lidar (16)
	adv_obs_normalizer = chainerrl.links.EmpiricalNormalization(
	action_space.low.size + 16, clip_threshold=5)

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

	adv_policy = chainer.Sequential(
				L.Linear(None, 128, initialW=winit), 
				F.tanh,
				L.Linear(None, 128, initialW=winit),
				F.tanh,
				L.Linear(None, action_size, initialW=winit_last),
				chainerrl.policies.GaussianHeadWithStateIndependentCovariance(
					action_size=action_size,
					var_type='diagonal',
					var_func=lambda x: F.exp(2 * x),  # Parameterize log std
					var_param_init=0,  # log std = 0 => std = 1
					),
				)

	adv_vf = chainer.Sequential(
				L.Linear(None, 256, initialW=winit),
				F.tanh,
				L.Linear(None, 64, initialW=winit),
				F.tanh,
				L.Linear(None, 1, initialW=winit),
				)
	
	model = chainerrl.links.Branched(policy, vf)
	adv_model = chainerrl.links.Branched(adv_policy, adv_vf)

	if args.gpu > -1:
		import cupy as cp
		model.to_gpu(args.gpu)
		adv_model.to_gpu(args.gpu)

	opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
	adv_opt = chainer.optimizers.Adam(alpha=args.lr, eps=1e-5)
	opt.setup(model)
	adv_opt.setup(adv_model)

	agent = PPO(model, opt,
		obs_normalizer=obs_normalizer, gpu=args.gpu,
		update_interval=args.update_interval, 
		minibatch_size=args.batchsize, epochs=10,
		clip_eps_vf=None, entropy_coef=0,
		standardize_advantages=args.standardize_advantages,
		gamma=0.995,
		lambd=0.97,
		)

	adversary = PPO(adv_model, adv_opt,
		obs_normalizer=adv_obs_normalizer, gpu=args.gpu,
		update_interval=2048, 
		minibatch_size=64, epochs=10,
		clip_eps_vf=None, entropy_coef=0,
		standardize_advantages=args.standardize_advantages,
		gamma=0.995,
		lambd=0.97,
		)

	if len(args.load) > 0:
		print('Loading trained agent')
		agent.load(args.load)
		print('Loading trained adversary')
		adversary.load(args.adv_load)

	#function to sample goals that are args.goal_norm away from nominal goal
	def gen_nonoverlap_goals(adv_goal):
		env_obs = env.reset()
		env_goal = env.goal_pos
		while  np.linalg.norm(adv_goal - env_goal) < args.goals_norm:
			env_obs = env.reset()
			env_goal = env.goal_pos
		return env_obs

	COLOR_BUTTON = np.array([0.8, .5, 0.2, 1])
	#initialize list to log statistics
	inference_stats = []
	
	print('Testing robust nominal policy trained with adversarial training from variant ' + str(args.variant))
	for i in range(10):
		_ = env.reset()
		#reset to generate random adversarial goal
		adv_goal = env.goal_pos
		#generate new nominal goal based on adv_goal
		env_obs = gen_nonoverlap_goals(adv_goal)
		last_dist_adv_goal = env.dist_xy(adv_goal)
		done = False
		t = 0.0
		env_R = 0.0
		adv_R = 0.0
		env_r = 0.0
		adv_r = 0.0
		nom_rate = 0
		adv_rate = 0
		
		while not done and t < 1000: 
			action = agent.act(env_obs)
			lidar_to_adv = env.obs_lidar([adv_goal],0)
			concat_obs = np.hstack((action, lidar_to_adv)).astype('float32')
			delta = adversary.act(concat_obs)
			adv_action = action + delta
			env_obs, env_r, done, _ = env.step(adv_action)
		
			# env.render()
			# env.render_area(adv_goal , 0.3, COLOR_BUTTON, 'adv_goal', alpha=0.5)

			dist_adv_goal = env.dist_xy(adv_goal)

			# penalty for adversarial policy going to nominal goal
			goal_penalty = 0
			if env.dist_xy(env.goal_pos) <= 0.35:
				goal_penalty = -1
				if env.dist_xy(adv_goal) <= 0.3:
					goal_penalty = 0

			#adversary reward function
			adv_r = (last_dist_adv_goal - dist_adv_goal)*1 + goal_penalty
			#manually scale rewards since env rewards not going through wrapper
			adv_r = adv_r*1e-2
			adv_r = adv_r.astype('float32')
			last_dist_adv_goal = dist_adv_goal

			#re-sample new pair of goals if nominal goal reached
			if  env.dist_xy(env.goal_pos) <= 0.35: 
				_ = env.reset()
				adv_goal = env.goal_pos
				env_obs = gen_nonoverlap_goals(adv_goal)
				last_dist_adv_goal = env.dist_xy(adv_goal)
				nom_rate += 1

			#re-sample new pair of goals if adversarial goal reached
			if  env.dist_xy(adv_goal) <= 0.3:
				adv_r += (1*1e-2)
				_ = env.reset()
				adv_goal = env.goal_pos
				env_obs = gen_nonoverlap_goals(adv_goal)
				last_dist_adv_goal = env.dist_xy(adv_goal)
				adv_rate += 1

			env_R += env_r
			adv_R += adv_r
			t += 1

		print('Episode:{}, Nom_Reward:{}, Adv_Reward:{}, Nom_Rate:{}, Adv_Rate:{}'.format(i, env_R, adv_R, nom_rate, adv_rate))
		inference_stats.append(np.asarray((i, env_R, adv_R, nom_rate, adv_rate)))
	
	inference_stats = np.asarray(inference_stats)
	if os.path.exists(args.save_dir) == False:
		os.makedirs(args.save_dir)
	save_path = args.save_dir + '/'  + args.env + '_' + str(args.variant) + '_' + str(args.goals_norm) + '_' + str(args.seed)
	np.save(save_path + '_'+ 'inference_stats.npy',inference_stats)
	
if __name__ == '__main__':
	main()