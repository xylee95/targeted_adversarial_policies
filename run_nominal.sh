steps=5000000
gpu=0
batchsize=1024

python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env PointGoal2.0-v1 --steps $steps --seed 0 
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env PointGoal2.0-v1 --steps $steps --seed 10
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env PointGoal2.0-v1 --steps $steps --seed 20
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env PointGoal2.0-v1 --steps $steps --seed 30
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env PointGoal2.0-v1 --steps $steps --seed 40

python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env CarGoal2.0-v1 --steps $steps --seed 0 
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env CarGoal2.0-v1 --steps $steps --seed 10
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env CarGoal2.0-v1 --steps $steps --seed 20
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env CarGoal2.0-v1 --steps $steps --seed 30
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env CarGoal2.0-v1 --steps $steps --seed 40