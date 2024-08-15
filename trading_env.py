import pandas as pd
import numpy as np

class TradingEnvironment:
    def __init__(self, prices, features=None, window_size=30, portfolio_ret_w=1.0, sharpe_w=0.1, drawdown_w=0.1, transaction_cost_w=0.1):
        self.prices = prices
        self.features = features if features is not None else prices
        self.window_size = window_size
        self.portfolio_ret_w = portfolio_ret_w
        self.sharpe_w = sharpe_w
        self.drawdown_w = drawdown_w
        self.transaction_cost_w = transaction_cost_w
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.done = False
        self.position = 0  # 1 for a long position, -1 for a short position, 0 for holding position
        self.cash = 1.0  # Starting with an initial cash of 1
        self.portfolio_value = 1.0
        self.portfolio_values = [1.0]
        self.position_change = 0
        return self._get_state()
    
    def _get_state(self):
        # state = self.prices[(self.current_step - self.window_size):self.current_step].values
        # return state
        start = self.current_step - self.window_size
        end = self.current_step
        state = self.features[start:end]
        if len(state.shape) == 1:
            state = state.reshape(-1, 1)
        return state

    def _calculate_reward(self):
        portfolio_return = (self.portfolio_value / self.portfolio_values[-1]) - 1.0
        sharpe_ratio = self._calculate_sharpe_ratio()
        drawdown = self._calculate_drawdown()
        transaction_costs = self._calculate_transaction_costs()
        reward = (self.portfolio_ret_w * portfolio_return) + (self.sharpe_w * sharpe_ratio) - (self.drawdown_w * drawdown) - (self.transaction_cost_w * transaction_costs)
        return reward

    def _calculate_sharpe_ratio(self):
        if len(self.portfolio_values) < 252:
            return 0
        daily_returns = pd.Series(self.portfolio_values).pct_change().dropna().values[-252:] # np.diff(self.portfolio_values[-252:]) / (self.portfolio_values[-253:-1])
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        if std_return == 0:
            return 0
        sharpe_ratio = np.sqrt(252) * (avg_return / std_return)    # annualized sharpe ratio
        return sharpe_ratio

    def _calculate_drawdown(self):
        if len(self.portfolio_values) < 252:
            return 0
        peak = np.maximum.accumulate(self.portfolio_values[-252:])
        drawdown = (peak - self.portfolio_values[-252:]) / peak
        return np.max(drawdown)

    def _calculate_transaction_costs(self):
        price = self.prices[self.current_step]
        trade_size = abs(self.position_change)
        trade_value = trade_size * price
        
        # commision (0.1% of trade, assuming no base commision)
        commission = 0.001 * trade_value
        
        # spread (0.1% of price from trade size)
        spread = (0.001 * trade_size) * price
        
        # slippage
        slippage = (0.0005 * trade_size) * price * trade_size   # slippage factor 0.05% for size 1, 0.1% for size 2 (increases with trade size)
        
        # market impact
        market_impact = (0.0005 * trade_size) * trade_value    # market impact factor 0.05% for size 1, 0.1% for size 2 (increases with trade size)
        
        total_cost = commission + spread + slippage + market_impact
        return total_cost

    def step(self, action):
        if self.done:
            raise Exception("Cannot step in a finished environment. Call reset to start a new episode.")

        self.current_step += 1
        
        if self.current_step >= len(self.prices):
            self.done = True
            return self._get_state(), 0, self.done, {}

        prev_position = self.position
        
        if action == 1:           # buy (go long)
            self.position = 1
        elif action == -1:        # sell (go short)
            self.position = -1
        else:                     # hold
            self.position = 0

        self.position_change = self.position - prev_position
        
        current_price = self.prices[self.current_step]
        transaction_cost = self._calculate_transaction_costs()
        self.cash -= transaction_cost
        
        if self.position_change > 0:      # buying
            self.cash -= abs(self.position_change) * current_price
        elif self.position_change < 0:    # selling
            self.cash += abs(self.position_change) * current_price
            
        self.portfolio_value = self.cash + (self.position * current_price)
        self.portfolio_values.append(self.portfolio_value)

        reward = self._calculate_reward()

        return self._get_state(), reward, self.done, {}
