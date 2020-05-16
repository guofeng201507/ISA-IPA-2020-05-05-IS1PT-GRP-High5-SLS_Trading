import gym
import json
import datetime as dt

import matplotlib.pyplot as plt

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2, DQN

from env.StockTradingEnvO import StockTradingEnv
# from stable_baselines.deepq.policies import MlpPolicy #only for dqn
# from env.StockDQNTradingEnv import StockTradingEnv #only for dqn

import pandas as pd
import os


plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 300

INITIAL_ACCOUNT_BALANCE = 50000

model_name = 'ppo2'


# dataset loading
df_train = pd.read_csv('./data/SPY_training.csv')
df_train = df_train.sort_values('Date')
df_test = pd.read_csv('./data/SPY_test.csv')
df_test = df_test.sort_values('Date')


# training
env = DummyVecEnv([lambda: StockTradingEnv(df_train)])
model = PPO2(MlpPolicy, env, verbose=1, seed=42, n_cpu_tf_sess=1, tensorboard_log="./tensorboard/")
# kwargs = {'double_q': False, 'prioritized_replay': False, 'policy_kwargs': dict(dueling=False)}
# model = DQN(MlpPolicy, env, verbose=1, tensorboard_log="./tensorboard/", **kwargs)
model.learn(total_timesteps=20000, log_interval=100)
# model.save(save_dir + model_name)

# del model 
# model = DQN.load(save_dir + model_name)

# testing
env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
obs = env.reset()
daily_profit = []
buy_hold_profit = []
for i in range(len(df_test)):
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)

    profit = env.render()
    daily_profit.append(profit)
    
    no_of_shares = INITIAL_ACCOUNT_BALANCE // df_test.loc[0, 'Close']
    
    buy_hold_profit_per_step = no_of_shares * (df_test.loc[i, 'Close'] - df_test.loc[0, 'Close'])
    buy_hold_profit.append(buy_hold_profit_per_step)


# plot result
fig, ax = plt.subplots()
ax.plot(daily_profit, '-o', label='Agent', marker='o', ms=2, alpha=0.7, mfc='orange')
ax.plot(buy_hold_profit, label='Market')
ax.grid()
plt.xlabel('step')
plt.ylabel('profit')
ax.legend()
plt.savefig(f'./img/test.png')
plt.show()
