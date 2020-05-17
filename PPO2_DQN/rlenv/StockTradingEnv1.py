import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 5000
MAX_OPEN_POSITIONS = 5
MAX_STEPS = 20000
MIN_COMMISSION = 1

INITIAL_ACCOUNT_BALANCE = 50000


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold, etc.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # Prices contains the OHCL values for the last five prices
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(11,), dtype=np.float16)

    def _next_observation(self):
        frame = np.array([
            self.df.loc[self.current_step, 'Open'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'High'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'Low'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'Close'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'Volume'] / MAX_NUM_SHARES,
        ])
        
        obs = np.append(frame, [
            self.balance / MAX_ACCOUNT_BALANCE,
            self.max_net_worth / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES,
            self.cost_basis / MAX_SHARE_PRICE,
            self.total_shares_sold / MAX_NUM_SHARES,
            self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        ], axis=0)


        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # current_price = random.uniform(
        #     self.df.loc[self.current_step, "Open"], self.df.loc[self.current_step, "Close"])
        
        # fixed current price to close
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
            
            self.commission_cost = commission_cost
            self.total_commission_cost += commission_cost
            self.balance -= additional_cost
            self.cost_basis = (
                prev_cost + additional_cost) / (self.shares_held + shares_bought)
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            if shares_sold > 0:
                commission_cost = max(MIN_COMMISSION,
                                      shares_sold * current_price * 0.001)  # Assume 0.1 % trade commission fee. min 1 dollar
            
            self.balance += shares_sold * current_price  - commission_cost
            self.commission_cost = commission_cost
            self.total_commission_cost += commission_cost
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the environment
        self._take_action(action)

        self.current_step += 1

        if self.current_step >= len(self.df.loc[:, 'Open'].values):
            self.current_step = 0
                  
        delay_modifier = (self.current_step / self.max_steps)
        
        # reward option 1.1
        reward = self.balance * delay_modifier
        
        # reward option 1.2
        # reward = self.net_worth
        
        # reward option 1.3
        # reward = (self.net_worth - INITIAL_ACCOUNT_BALANCE) * delay_modifier  # profit
        
        # reward option 2
        # if self.current_step == self.share_start_step+1:
        #     self.new_shares_held = 0
        #     self.old_shares_held = 0
        # else:
        #     self.old_shares_held = self.new_shares_held
        #     self.new_shares_held = self.shares_held
            
        # if self.new_shares_held == self.old_shares_held:
        #     self.same_shares_held += 1
        #     if self.same_shares_held >5:
        #         reward = - self.same_shares_held
        
        # reward option 3
        # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE - 5 # profit
        
        # reward option 4
        # if reward > 0:
        #     reward = 1
        # elif reward == 0:
        #     reward = -100
        # else:
        #     reward = -100
        
        done = self.net_worth <= 0

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0
        self.same_shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0
        self.total_commission_cost = 0
        self.commission_cost = 0
        self.max_steps = len(self.df)

        self.current_step = 0
        self.share_start_step = self.current_step
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Total commission cost: {self.total_commission_cost}')
        print(f'Profit: {profit}')
        return profit