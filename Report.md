# Report
---

## Learning Algorithm

For the solution the Multi-Agent Deep Deterministic Policy Gradient (MADDPG) will be used. MADDPG is a multi-agent version of DDPG (defined in (`ddpg_agent.py`). 
For MADDPG, each agent create it's own actor and it's own critic. Agents share a common experience replay buffer which contains  states and actions from all agents,so that  each agent doessampling from this replay buffer. 

The  environment is considered solved when the average reward (over the last 100 episodes) is at least +0.5. Note the maximum reward for both of the agents is used for each time step as the reward. 
### MADDPG Hyper Parameters
- n_episodes (int): maximum number of training episodes
- max_t (int): maximum number of timesteps per episode (increase for exploring more states) :1000
- num_agents: number of agents in the environment : 2

- BUFFER_SIZE (int): replay buffer size :1e5
- BATCH_SIZE (int): mini batch size : 512
- GAMMA (float): discount factor : 0.99 
- TAU (float): for soft update of target parameters : 1e-2 
- LR_ACTOR (float): learning rate for optimizer : 1e-4 
- LR_CRITIC (float): learning rate for optimizer : 3e-4 
- WEIGHT_DECAY (float): L2 weight decay  : 0.0
- N_TIME_STEPS (int): every n time step do update : 8
- NOISE : Ornstein-Uhlenbeck Noise : 2
- NOISE_REDUCTION : Noise redusction : 0.9999

### Neural Networks
Actor and Critic  models are defines in `ddpg_model.py`.

The Actor networks has  three fully connected layers with  relu activation and tanh activation for the action space. 
  - input: state size  output 256
  - input: 256  output 128
   - input: 128  output action size 

The Critic networks has  three fully connected layers with leaky_relu activation. 
  - input: state size  output 256
  - input: 256  output 128
   - input: 128  output 1

## Plot of rewards
![Reward Plot]() 
![Reward Plot]()

```
Episode 100	Average Score: 0.00.72	Rewards [ 0.00 -0.01]	Scores: [ 0.00 -0.01]
Episode 200	Average Score: 0.00.49	Rewards [-0.01  0.00]	Scores: [-0.01  0.00]
Episode 300	Average Score: 0.00.29	Rewards [ 0.00 -0.01]	Scores: [ 0.00 -0.01]
Episode 400	Average Score: 0.01.09	Rewards [-0.01  0.00]	Scores: [-0.01  0.00]
Episode 500	Average Score: 0.01.92	Rewards [ 0.00 -0.01]	Scores: [ 0.00 -0.01]
Episode 600	Average Score: 0.04.72	Rewards [-0.01  0.00]	Scores: [-0.01  0.00]
Episode 700	Average Score: 0.06.53	Rewards [ 0.00 -0.01]	Scores: [ 0.20  0.09]
Episode 800	Average Score: 0.07.38	Rewards [-0.01  0.00]	Scores: [-0.01  0.00]]
Episode 900	Average Score: 0.25.14	Rewards [ 0.00 -0.01]	Scores: [ 0.10  0.09]]
Episode 925	Timestep 1000	Noise 0.04	Rewards [ 0.00  0.00]	Scores: [ 2.60  2.60]
Environment solved in 925 episodes!	Average Score: 0.52
Episode 1000	Average Score: 1.73 0.00	Rewards [ 0.00  0.00]	Scores: [ 2.70  2.60]
Episode 1050	Timestep 1000	Noise 0.00	Rewards [ 0.00  0.00]	Scores: [ 2.60  2.60]
```

## Improvements
- I believe that Proximal Policy Optimization (PPO) and Distributed Distributional Deterministic Policy Gradients (D4PG) can improve our score.

- Fine tuning on the hyper parameters can bring a good result.


