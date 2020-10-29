# Test nominal goal success rate 
goals_norm=1.0
gpu=0
env=PointGoal2.0-v1
python nominal_inference.py --gpu $gpu --seed 0 --env $env  --load nominal/PointGoal2.0-v1_0  --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 10 --env $env --load nominal/PointGoal2.0-v1_10 --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 20 --env $env --load nominal/PointGoal2.0-v1_20 --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 30 --env $env --load nominal/PointGoal2.0-v1_30 --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 40 --env $env --load nominal/PointGoal2.0-v1_40 --goals_norm $goals_norm

env=CarGoal2.0-v1
python nominal_inference.py --gpu $gpu --seed 0 --env $env  --load nominal/PointGoal2.0-v1_0  --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 10 --env $env --load nominal/PointGoal2.0-v1_10 --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 20 --env $env --load nominal/PointGoal2.0-v1_20 --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 30 --env $env --load nominal/PointGoal2.0-v1_30 --goals_norm $goals_norm
python nominal_inference.py --gpu $gpu --seed 40 --env $env --load nominal/PointGoal2.0-v1_40 --goals_norm $goals_norm
