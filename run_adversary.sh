#!/bin/bash
steps=5000000
gpu=0
python train_adversarial_agent.py --gpu $gpu --seed 0  --env PointGoal2.0-v1 --variant A --load nominal/weights/PointGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env PointGoal2.0-v1 --variant A --load nominal/weights/PointGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env PointGoal2.0-v1 --variant A --load nominal/weights/PointGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env PointGoal2.0-v1 --variant A --load nominal/weights/PointGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env PointGoal2.0-v1 --variant A --load nominal/weights/PointGoal2.0-v1_40 --steps $steps

python train_adversarial_agent.py --gpu $gpu --seed 0  --env PointGoal2.0-v1 --variant SA --load nominal/weights/PointGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env PointGoal2.0-v1 --variant SA --load nominal/weights/PointGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env PointGoal2.0-v1 --variant SA --load nominal/weights/PointGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env PointGoal2.0-v1 --variant SA --load nominal/weights/PointGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env PointGoal2.0-v1 --variant SA --load nominal/weights/PointGoal2.0-v1_40 --steps $steps

python train_adversarial_agent.py --gpu $gpu --seed 0  --env CarGoal2.0-v1 --variant A --load nominal/weights/CarGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env CarGoal2.0-v1 --variant A --load nominal/weights/CarGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env CarGoal2.0-v1 --variant A --load nominal/weights/CarGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env CarGoal2.0-v1 --variant A --load nominal/weights/CarGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env CarGoal2.0-v1 --variant A --load nominal/weights/CarGoal2.0-v1_40 --steps $steps

python train_adversarial_agent.py --gpu $gpu --seed 0  --env CarGoal2.0-v1 --variant SA --load nominal/weights/CarGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env CarGoal2.0-v1 --variant SA --load nominal/weights/CarGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env CarGoal2.0-v1 --variant SA --load nominal/weights/CarGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env CarGoal2.0-v1 --variant SA --load nominal/weights/CarGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env CarGoal2.0-v1 --variant SA --load nominal/weights/CarGoal2.0-v1_40 --steps $steps