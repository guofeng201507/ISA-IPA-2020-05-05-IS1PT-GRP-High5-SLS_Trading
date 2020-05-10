import pandas as pd
from stable_baselines import PPO2
from stable_baselines.common.vec_env import DummyVecEnv

from rlenv.StockTradingEnv_US import StockTradingEnv_US

INITIAL_ACCOUNT_BALANCE = 50000


def run_PPO2_model(test_data_file, model_path):
    model = PPO2.load(model_path)

    day_profits = []
    buy_hold_profit = []

    df_test = pd.read_csv(test_data_file)

    env_test = DummyVecEnv([lambda: StockTradingEnv_US(df_test)])

    obs = env_test.reset()
    no_of_shares = 0
    buy_hold_commission = 0
    for i in range(len(df_test) - 1):
        if i > 22:
            break
        action, _states = model.predict(obs)
        obs, rewards, done, info = env_test.step(action)
        profit = env_test.render()
        day_profits.append(profit)
        if i == 0:
            buy_hold_profit.append(0)
            no_of_shares = INITIAL_ACCOUNT_BALANCE // df_test.iloc[0]['Close']
            buy_hold_commission = no_of_shares * df_test.iloc[0]['Close'] * 0.001
            print('Buy ' + str(no_of_shares) + ' shares and hold')
        else:
            buy_hold_profit_per_step = no_of_shares * (
                    df_test.iloc[i]['Close'] - df_test.iloc[0]['Close']) - buy_hold_commission
            buy_hold_profit.append(buy_hold_profit_per_step)
            print('Buy and Hold: ' + '*' * 40)
            print('No of shares: ' + str(no_of_shares) + ' average cost per share ' + str(df_test.iloc[0]['Close']))
            print('profit is ' + str(buy_hold_profit_per_step))


run_PPO2_model('./stockdata/test/QQQ_New_test.csv', './model/model_QQQ_50000_0.dat')
