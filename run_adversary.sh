#!/bin/bash
steps=5000000
gpu=0
variant=A 
env=PointGoal2.0-v1
python train_adversarial_agent.py --gpu $gpu --seed 0  --env $env --variant $variant --load nominal/PointGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env $env --variant $variant --load nominal/PointGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env $env --variant $variant --load nominal/PointGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env $env --variant $variant --load nominal/PointGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env $env --variant $variant --load nominal/PointGoal2.0-v1_40 --steps $steps

variant=SA 
env=PointGoal2.0-v1
python train_adversarial_agent.py --gpu $gpu --seed 0  --env $env --variant $variant --load nominal/PointGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env $env --variant $variant --load nominal/PointGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env $env --variant $variant --load nominal/PointGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env $env --variant $variant --load nominal/PointGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env $env --variant $variant --load nominal/PointGoal2.0-v1_40 --steps $steps

variant=A 
env=CarGoal2.0-v1
python train_adversarial_agent.py --gpu $gpu --seed 0  --env $env --variant $variant --load nominal/CarGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env $env --variant $variant --load nominal/CarGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env $env --variant $variant --load nominal/CarGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env $env --variant $variant --load nominal/CarGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env $env --variant $variant --load nominal/CarGoal2.0-v1_40 --steps $steps

variant=SA 
env=CarGoal2.0-v1
python train_adversarial_agent.py --gpu $gpu --seed 0  --env $env --variant $variant --load nominal/CarGoal2.0-v1_0  --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 10 --env $env --variant $variant --load nominal/CarGoal2.0-v1_10 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 20 --env $env --variant $variant --load nominal/CarGoal2.0-v1_20 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 30 --env $env --variant $variant --load nominal/CarGoal2.0-v1_30 --steps $steps
python train_adversarial_agent.py --gpu $gpu --seed 40 --env $env --variant $variant --load nominal/CarGoal2.0-v1_40 --steps $steps
