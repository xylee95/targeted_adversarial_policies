gpu=0
goals_norm=0.5
 
#Example: evaluate robust trained agent trained with adversarial training variant 3 under attack by adversarial policy variant (A)

python robust_inference.py --variant 3 --gpu $gpu --seed 0  --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_3_0  --adv_load adversary/weights/PointGoal2.0-v1_A_0 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 10 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_3_10 --adv_load adversary/weights/PointGoal2.0-v1_A_10 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 20 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_3_20 --adv_load adversary/weights/PointGoal2.0-v1_A_20 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 30 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_3_30 --adv_load adversary/weights/PointGoal2.0-v1_A_30 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 40 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_3_40 --adv_load adversary/weights/PointGoal2.0-v1_A_40 --goals_norm $goals_norm

# Variants 1 and 2 not effective to robustify agent as shown in paper
python robust_inference.py --variant 2 --gpu $gpu --seed 0  --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_2_0  --adv_load adversary/weights/PointGoal2.0-v1_A_0 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 10 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_2_10 --adv_load adversary/weights/PointGoal2.0-v1_A_10 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 20 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_2_20 --adv_load adversary/weights/PointGoal2.0-v1_A_20 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 30 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_2_30 --adv_load adversary/weights/PointGoal2.0-v1_A_30 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 40 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_2_40 --adv_load adversary/weights/PointGoal2.0-v1_A_40 --goals_norm $goals_norm

python robust_inference.py --variant 1 --gpu $gpu --seed 0  --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_1_0  --adv_load adversary/weights/PointGoal2.0-v1_A_0 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 10 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_1_10 --adv_load adversary/weights/PointGoal2.0-v1_A_10 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 20 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_1_20 --adv_load adversary/weights/PointGoal2.0-v1_A_20 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 30 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_1_30 --adv_load adversary/weights/PointGoal2.0-v1_A_30 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 40 --env PointGoal2.0-v1 --load robust/weights/PointGoal2.0-v1_1_40 --adv_load adversary/weights/PointGoal2.0-v1_A_40 --goals_norm $goals_norm

#For CarGoal environment
python robust_inference.py --variant 3 --gpu $gpu --seed 0  --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_3_0  --adv_load adversary/weights/CarGoal2.0-v1_A_0 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 10 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_3_10 --adv_load adversary/weights/CarGoal2.0-v1_A_10 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 20 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_3_20 --adv_load adversary/weights/CarGoal2.0-v1_A_20 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 30 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_3_30 --adv_load adversary/weights/CarGoal2.0-v1_A_30 --goals_norm $goals_norm
# python robust_inference.py --variant 3 --gpu $gpu --seed 40 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_3_40 --adv_load adversary/weights/CarGoal2.0-v1_A_40 --goals_norm $goals_norm

# Variants 1 and 2 not effective to robustify agent as shown in paper
python robust_inference.py --variant 2 --gpu $gpu --seed 0  --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_2_0  --adv_load adversary/weights/CarGoal2.0-v1_A_0 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 10 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_2_10 --adv_load adversary/weights/CarGoal2.0-v1_A_10 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 20 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_2_20 --adv_load adversary/weights/CarGoal2.0-v1_A_20 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 30 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_2_30 --adv_load adversary/weights/CarGoal2.0-v1_A_30 --goals_norm $goals_norm
# python robust_inference.py --variant 2 --gpu $gpu --seed 40 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_2_40 --adv_load adversary/weights/CarGoal2.0-v1_A_40 --goals_norm $goals_norm

python robust_inference.py --variant 1 --gpu $gpu --seed 0  --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_1_0  --adv_load adversary/weights/CarGoal2.0-v1_A_0 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 10 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_1_10 --adv_load adversary/weights/CarGoal2.0-v1_A_10 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 20 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_1_20 --adv_load adversary/weights/CarGoal2.0-v1_A_20 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 30 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_1_30 --adv_load adversary/weights/CarGoal2.0-v1_A_30 --goals_norm $goals_norm
# python robust_inference.py --variant 1 --gpu $gpu --seed 40 --env CarGoal2.0-v1 --load robust/weights/CarGoal2.0-v1_1_40 --adv_load adversary/weights/CarGoal2.0-v1_A_40 --goals_norm $goals_norm