import os
import pickle

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from rlenv.StockTradingEnv_US import StockTradingEnv_US

# from stable_baselines.deepq.policies import MlpPolicy

font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False

INITIAL_ACCOUNT_BALANCE = 50000


def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade_US(stock_code, counter):
    stock_file_train = find_file('./stockdata/train', str(stock_code))

    NO_OF_TEST_TRADING_DAYS = 104

    daily_profits, buy_hold_profit, good_model, model, total_steps = stock_trade_US(stock_file_train,
                                                                                    NO_OF_TEST_TRADING_DAYS)
    if good_model:
        model.save(f'./model/model_{stock_code}_{total_steps}_{counter}.dat')

    fig, ax = plt.subplots()
    ax.plot(daily_profits, '-o', label='AI SPY', marker='o', ms=2, alpha=0.7, mfc='orange')
    ax.plot(buy_hold_profit, '-o', label='B&H SPY', marker='o', ms=2, alpha=0.7, mfc='blue')
    ax.grid()
    plt.xlabel('step')
    plt.ylabel('profit')
    ax.legend(prop=font)
    # plt.show()
    plt.savefig(f'./img/{stock_code}_{total_steps}_days_{NO_OF_TEST_TRADING_DAYS}_new.png')


def multi_stock_trade():
    start_code = 600000
    max_num = 3000

    group_result = []

    for code in range(start_code, start_code + max_num):
        stock_file = find_file('./stockdata/train', str(code))
        if stock_file:
            try:
                profits = stock_trade_US(stock_file)
                group_result.append(profits)
            except Exception as err:
                print(err)

    with open(f'code-{start_code}-{start_code + max_num}.pkl', 'wb') as f:
        pickle.dump(group_result, f)


def stock_trade_US(stock_file_train, no_of_test_trading_days):
    df_train = pd.read_csv(stock_file_train)
    # df_train = df_train.sort_values('date')

    # The algorithms require a vectorized environment to run
    env_train = DummyVecEnv([lambda: StockTradingEnv_US(df_train)])

    total_timesteps = int(5e4)
    # total_timesteps = int(1e5)

    model = PPO2('MlpPolicy', env_train, verbose=0, tensorboard_log='./log', seed=12345).learn(
        total_timesteps=total_timesteps)

    # Random Agent, after training
    # mean_reward, std_reward = evaluate_policy(model, env_train, n_eval_episodes=100)
    # print(f"after training, mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")

    # -----------------Test Model --------------------------------------

    import sys
    sys.stdout = open(f'./output/output_SPY_{total_timesteps}_days_{no_of_test_trading_days}.txt', 'wt')

    day_profits = []
    buy_hold_profit = []

    df_test = pd.read_csv(stock_file_train.replace('train', 'test')).drop(['Adj Close'], axis=1)

    env_test = DummyVecEnv([lambda: StockTradingEnv_US(df_test)])
    obs = env_test.reset()
    no_of_shares = 0
    buy_hold_commission = 0
    for n in range(len(df_test) - 1):
        if n > no_of_test_trading_days:
            break

        action, _states = model.predict(obs)

        # let agent start with a buy all
        # if n == 0:
        #     action[0][0] = 0
        #     action[0][1] = 1

        obs, rewards, done, info = env_test.step(action)
        profit = env_test.render()
        day_profits.append(profit)

        if n == 0:
            buy_hold_profit.append(0)
            no_of_shares = INITIAL_ACCOUNT_BALANCE // df_test.iloc[0]['Close']
            buy_hold_commission = no_of_shares * df_test.iloc[0]['Close'] * 0.001
            print('Buy ' + str(no_of_shares) + ' shares and hold')
        else:
            buy_hold_profit_per_step = no_of_shares * (
                    df_test.iloc[n]['Close'] - df_test.iloc[0]['Close']) - buy_hold_commission
            buy_hold_profit.append(buy_hold_profit_per_step)
            print('Buy and Hold: ' + '*' * 40)
            print('No of shares: ' + str(no_of_shares) + ' average cost per share ' + str(df_test.iloc[0]['Close']))
            print('profit is ' + str(buy_hold_profit_per_step))

        if done:
            break

    good_model = False
    if day_profits[-1] > buy_hold_profit[-1]:
        good_model = True

    return day_profits, buy_hold_profit, good_model, model, total_timesteps


if __name__ == '__main__':
    # multi_stock_trade()
    test_a_stock_trade_US('SPY', 0)

    # for i in range(1, 10):
    #     print(f'Iteration {i} ')
    #     test_a_stock_trade_US('SPY', i)
    # ret = find_file('./stockdata/train', '600036')
    # print(ret)
