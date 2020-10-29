steps=5000000
gpu=0
batchsize=1024

env=PointGoal2.0-v1
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 0 
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 10
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 20
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 30
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 40

env=CarGoal2.0-v1
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 0 
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 10
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 20
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 30
python train_nominal_agent.py --gpu $gpu --batchsize $batchsize --env $env --steps $steps --seed 40