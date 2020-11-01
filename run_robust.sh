#!/bin/bash
steps=5000
gpu=0
#variant 1 of adversarial training, no pre-trained weights used
python train_robust_agent.py --gpu $gpu --seed 0  --env PointGoal2.0-v1 --variant 1  --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 10 --env PointGoal2.0-v1 --variant 1 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 20 --env PointGoal2.0-v1 --variant 1 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 30 --env PointGoal2.0-v1 --variant 1 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 40 --env PointGoal2.0-v1 --variant 1 --steps $steps

#variant 2 of adv. training, only load adv. weights
python train_robust_agent.py --gpu $gpu --seed 0  --env PointGoal2.0-v1 --variant 2 --adv_load adversary/weights/PointGoal2.0-v1_A_0  --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 10 --env PointGoal2.0-v1 --variant 2 --adv_load adversary/weights/PointGoal2.0-v1_A_10 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 20 --env PointGoal2.0-v1 --variant 2 --adv_load adversary/weights/PointGoal2.0-v1_A_20 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 30 --env PointGoal2.0-v1 --variant 2 --adv_load adversary/weights/PointGoal2.0-v1_A_30 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 40 --env PointGoal2.0-v1 --variant 2 --adv_load adversary/weights/PointGoal2.0-v1_A_40 --steps $steps

#v3 of adv. training, load both agent's weights
python train_robust_agent.py --gpu $gpu --seed 0  --env PointGoal2.0-v1 --variant 3 --load nominal/weights/PointGoal2.0-v1_0 --adv_load adversary/weights/PointGoal2.0-v1_A_0  --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 10 --env PointGoal2.0-v1 --variant 3 --load nominal/weights/PointGoal2.0-v1_10 --adv_load adversary/weights/PointGoal2.0-v1_A_10 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 20 --env PointGoal2.0-v1 --variant 3 --load nominal/weights/PointGoal2.0-v1_20 --adv_load adversary/weights/PointGoal2.0-v1_A_20 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 30 --env PointGoal2.0-v1 --variant 3 --load nominal/weights/PointGoal2.0-v1_30 --adv_load adversary/weights/PointGoal2.0-v1_A_30 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 40 --env PointGoal2.0-v1 --variant 3 --load nominal/weights/PointGoal2.0-v1_40 --adv_load adversary/weights/PointGoal2.0-v1_A_40 --steps $steps

#variant 1 of adversarial training, no pre-trained weights used
python train_robust_agent.py --gpu $gpu --seed 0  --env CarGoal2.0-v1 --variant 1  --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 10 --env CarGoal2.0-v1 --variant 1 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 20 --env CarGoal2.0-v1 --variant 1 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 30 --env CarGoal2.0-v1 --variant 1 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 40 --env CarGoal2.0-v1 --variant 1 --steps $steps

#variant 2 of adv. training, only load adv. weights
python train_robust_agent.py --gpu $gpu --seed 0  --env CarGoal2.0-v1 --variant 2 --adv_load adversary/weights/CarGoal2.0-v1_A_0  --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 10 --env CarGoal2.0-v1 --variant 2 --adv_load adversary/weights/CarGoal2.0-v1_A_10 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 20 --env CarGoal2.0-v1 --variant 2 --adv_load adversary/weights/CarGoal2.0-v1_A_20 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 30 --env CarGoal2.0-v1 --variant 2 --adv_load adversary/weights/CarGoal2.0-v1_A_30 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 40 --env CarGoal2.0-v1 --variant 2 --adv_load adversary/weights/CarGoal2.0-v1_A_40 --steps $steps

#v3 of adv. training, load both agent's weights
python train_robust_agent.py --gpu $gpu --seed 0  --env CarGoal2.0-v1 --variant 3 --load nominal/weights/CarGoal2.0-v1_0 --adv_load adversary/weights/CarGoal2.0-v1_A_0  --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 10 --env CarGoal2.0-v1 --variant 3 --load nominal/weights/CarGoal2.0-v1_10 --adv_load adversary/weights/CarGoal2.0-v1_A_10 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 20 --env CarGoal2.0-v1 --variant 3 --load nominal/weights/CarGoal2.0-v1_20 --adv_load adversary/weights/CarGoal2.0-v1_A_20 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 30 --env CarGoal2.0-v1 --variant 3 --load nominal/weights/CarGoal2.0-v1_30 --adv_load adversary/weights/CarGoal2.0-v1_A_30 --steps $steps
# python train_robust_agent.py --gpu $gpu --seed 40 --env CarGoal2.0-v1 --variant 3 --load nominal/weights/CarGoal2.0-v1_40 --adv_load adversary/weights/CarGoal2.0-v1_A_40 --steps $steps
