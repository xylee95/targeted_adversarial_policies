# Targeted Action Space Adversarial Policy on Deep RL Agents
## Implementation
Code implementation from the paper "Query-based Targeted Action-Space Adversarial Policies on Deep Reinforcement Learning Agents".  

### Pre-requisites 
The main packages required are ChainerRL, Safety-Gym and MuJoCo. For a complete list of required packages, please refer to env.yaml. 

### Code Structure
1. train_nominal_agent.py: code to train nominal policy in environment with PPO
2. nominal_inference.py: code to evaluate nominal policy performance
3. train_adversary_agent.py: code to train adversarial policy on top of nominal policy for targeted attacks
4. adversary_inference.py: code to evaluate adversarial policy performance
5. train_robust_agent.py: code to adversarially train nominal policy in presence of targeted adversarial attacks
6. robust_inferece.py: code to evaluate robust policy performance
7. shell files: scripts to run experiments for multiple seeds and various parameters

### Special arguments to take note
1. train_adversary_agent.py can be run with two variants
    * --variant **SA** trains adversarial policy that observes nominal policy's states and actions (StateAware)
    * --variant **A**  trains adversarial policy that observes only nominal policy's actions (StateUnware)
2. train_robust_agent.py can be run with three variants
    * --variant **1**  train nominal and adversarial policy in parallel with random weight initializations
    * --variant **2**  train nominal policy in the presence of a trained adversary policy 
    * --variant **3**  train nominal policy in the presence of a trained adversary policy, but initialize nominal policy with previously trained weights
