gpu=0
goals_norm=0.5
python adversary_inference.py --variant A --gpu $gpu --seed 0  --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_0  --adv_load adversary/weights/PointGoal2.0-v1_A_0 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 10 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_10 --adv_load adversary/weights/PointGoal2.0-v1_A_10 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 20 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_20 --adv_load adversary/weights/PointGoal2.0-v1_A_20 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 30 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_30 --adv_load adversary/weights/PointGoal2.0-v1_A_30 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 40 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_40 --adv_load adversary/weights/PointGoal2.0-v1_A_40 --goals_norm $goals_norm

python adversary_inference.py --variant SA --gpu $gpu --seed 0  --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_0  --adv_load adversary/weights/PointGoal2.0-v1_SA_0 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 20 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_20 --adv_load adversary/weights/PointGoal2.0-v1_SA_20 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 30 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_30 --adv_load adversary/weights/PointGoal2.0-v1_SA_30 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 40 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_40 --adv_load adversary/weights/PointGoal2.0-v1_SA_40 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 10 --env PointGoal2.0-v1 --load nominal/weights/PointGoal2.0-v1_10 --adv_load adversary/weights/PointGoal2.0-v1_SA_10 --goals_norm $goals_norm

python adversary_inference.py --variant A --gpu $gpu --seed 0  --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_0  --adv_load adversary/weights/CarGoal2.0-v1_A_0 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 10 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_10 --adv_load adversary/weights/CarGoal2.0-v1_A_10 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 20 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_20 --adv_load adversary/weights/CarGoal2.0-v1_A_20 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 30 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_30 --adv_load adversary/weights/CarGoal2.0-v1_A_30 --goals_norm $goals_norm
python adversary_inference.py --variant A --gpu $gpu --seed 40 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_40 --adv_load adversary/weights/CarGoal2.0-v1_A_40 --goals_norm $goals_norm

python adversary_inference.py --variant SA --gpu $gpu --seed 0  --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_0  --adv_load adversary/weights/CarGoal2.0-v1_SA_0 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 20 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_20 --adv_load adversary/weights/CarGoal2.0-v1_SA_20 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 10 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_10 --adv_load adversary/weights/CarGoal2.0-v1_SA_10 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 30 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_30 --adv_load adversary/weights/CarGoal2.0-v1_SA_30 --goals_norm $goals_norm
python adversary_inference.py --variant SA --gpu $gpu --seed 40 --env CarGoal2.0-v1 --load nominal/weights/CarGoal2.0-v1_40 --adv_load adversary/weights/CarGoal2.0-v1_SA_40 --goals_norm $goals_norm