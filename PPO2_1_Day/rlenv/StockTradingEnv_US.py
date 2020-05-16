import gym
import numpy as np
from gym import spaces
from random import random

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_VOLUME = 1000e8
MAX_AMOUNT = 3e10
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MAX_DAY_CHANGE = 1
MIN_COMMISSION = 1

INITIAL_ACCOUNT_BALANCE = 50000


class StockTradingEnv_US(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv_US, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float16)

    def _next_observation(self):
        obs = np.array([
            self.df.loc[self.current_step, 'Open'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'High'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'Low'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'Close'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'Volume'] / MAX_VOLUME,
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.commission_cost / MAX_ACCOUNT_BALANCE,
            self.total_commission_cost / MAX_ACCOUNT_BALANCE,
            self.shares_bought / MAX_NUM_SHARES,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ])
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # current_price = random.uniform(
        #     self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])
        # self.current_step = random.randint(0, len(self.df.loc[:, 'Open'].values) - 6)
        current_price = self.df.loc[self.current_step, "Close"]

        action_type = action[0]
        amount = action[1]
        commission_cost = 0

        if action_type < 1:
            # Buy amount % of balance in shares
            total_possible = int(self.balance / current_price)
            shares_bought = int(total_possible * amount)
            prev_cost = self.cost_basis * self.shares_held
            additional_cost = shares_bought * current_price
            if shares_bought > 0:
                commission_cost = max(MIN_COMMISSION, additional_cost * 0.001)  # Assume 0.1 % trade commission fee
            additional_cost += commission_cost

            self.shares_bought = shares_bought
            self.shares_sold = 0
            self.commission_cost = commission_cost
            self.total_commission_cost += commission_cost
            self.balance -= additional_cost
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            if shares_sold > 0:
                commission_cost = max(MIN_COMMISSION,
                                      shares_sold * current_price * 0.001)  # Assume 0.1 % trade commission fee. min 1 dollar
            self.balance += shares_sold * current_price - commission_cost

            self.commission_cost = commission_cost
            self.total_commission_cost += commission_cost
            self.shares_held -= shares_sold
            self.shares_bought = 0
            self.shares_sold = shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)
        done = False

        self.current_step += 1

        if self.current_step > len(self.df.loc[:, 'Open'].values) - 1:
            self.current_step = 0  # loop training
            # done = True

        delay_modifier = (self.current_step / MAX_STEPS)

        # profits
        reward = self.net_worth - INITIAL_ACCOUNT_BALANCE

        # reward = self.net_worth - 248 * self.df.loc[self.current_step, "Close"] #compare with buy & hold
        if reward > 0:
            reward = 1
        elif reward == 0:
            reward = -100
        else:
            reward = -100

        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.cost_basis = 0
        self.shares_bought = 0
        self.shares_sold = 0
        self.total_sales_value = 0
        self.total_commission_cost = 0
        self.commission_cost = 0

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-' * 30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')

        print(f'Shares bought : {self.shares_bought}')
        print(f'Shares sold : {self.shares_sold}')
        print(f'Shares held: {self.shares_held} ')

        print(f'Total commission cost: {self.total_commission_cost}')
        print(f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')

        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        return profit
