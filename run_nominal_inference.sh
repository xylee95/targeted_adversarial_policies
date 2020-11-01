# Test nominal goal success rate 
goals_norm=1.0
gpu=0
python nominal_inference.py --gpu $gpu --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_0 --goals_norm $goals_norm --seed 0
python nominal_inference.py --gpu $gpu --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_10 --goals_norm $goals_norm --seed 10
python nominal_inference.py --gpu $gpu --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_20 --goals_norm $goals_norm --seed 20
python nominal_inference.py --gpu $gpu --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_30 --goals_norm $goals_norm --seed 30
python nominal_inference.py --gpu $gpu --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_40 --goals_norm $goals_norm --seed 40

python nominal_inference.py --gpu $gpu --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_0 --goals_norm $goals_norm --seed 0
python nominal_inference.py --gpu $gpu --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_10 --goals_norm $goals_norm --seed 0
python nominal_inference.py --gpu $gpu --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_20 --goals_norm $goals_norm --seed 0
python nominal_inference.py --gpu $gpu --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_30 --goals_norm $goals_norm --seed 0
python nominal_inference.py --gpu $gpu --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_40 --goals_norm $goals_norm --seed 0
