import gym
import numpy as np
from gym import spaces

class SoccerBetEnv(gym.Env):
    def __init__(self):
        super(SoccerBetEnv, self).__init__()
        
        # Define the discrete action space (0: No Bet, 1-15: Different bets)
        self.action_space = spaces.Discrete(16)
        
        # Observation space: odds, time left, score, remaining bets/budget, expected returns
        self.observation_space = spaces.Box(
            low=np.array([1.01, 1.01, 1.01, 0, 0, 0, 0, 0, -1000, -1000, -1000]),
            high=np.array([50, 50, 50, 90, 10, 10, 5, 1000, 1000, 1000, 1000]),
            dtype=np.float32
        )
        
        self.reset()

    def reset(self):
        """Reset the environment for a new match"""
        self.odds = np.random.uniform(1.5, 5.0, 3)  # Random initial odds
        self.minutes_left = 90
        self.score_A, self.score_B = 0, 0
        self.bets_remaining = 5  # Max bets allowed
        self.budget = 100  # Fixed budget per game
        self.expected_return_A, self.expected_return_Draw, self.expected_return_B = 0, 0, 0
        
        return self._get_obs()
    
    def _get_obs(self):
        """Return the current state"""
        return np.array([
            *self.odds, self.minutes_left, self.score_A, self.score_B,
            self.bets_remaining, self.budget, self.expected_return_A,
            self.expected_return_Draw, self.expected_return_B
        ], dtype=np.float32)
    
    def step(self, action):
        """Take an action (place a bet or do nothing)"""
        reward = 0
        done = False
        stake_options = [0, 0.1, 0.2, 0.5, 0.7, 1.0]  # Betting percentages
        
        if action > 0 and self.bets_remaining > 0 and self.budget > 0:
            outcome_idx = (action - 1) // 5  # 0: Team A, 1: Draw, 2: Team B
            stake_percentage = stake_options[(action - 1) % 5 + 1]
            stake = self.budget * stake_percentage  # Calculate stake
            
            self.bets_remaining -= 1
            self.budget -= stake
            
            # Simulate match outcome randomly (to be replaced with real results)
            match_result = np.random.choice([0, 1, 2])  # 0: A wins, 1: Draw, 2: B wins
            
            if match_result == outcome_idx:
                profit = (stake * self.odds[outcome_idx]) - stake
                reward = profit
                if outcome_idx == 0:
                    self.expected_return_A += profit
                elif outcome_idx == 1:
                    self.expected_return_Draw += profit
                else:
                    self.expected_return_B += profit
            else:
                reward = -stake
                if outcome_idx == 0:
                    self.expected_return_A -= stake
                elif outcome_idx == 1:
                    self.expected_return_Draw -= stake
                else:
                    self.expected_return_B -= stake
        
        # Update time & simulate minor odds movements
        self.minutes_left -= 1
        self.odds *= np.random.uniform(0.98, 1.02, 3)
        
        # Ensure odds remain within limits
        self.odds = np.clip(self.odds, 1.01, 50)
        
        if self.minutes_left <= 0 or self.bets_remaining == 0:
            done = True  # End of game
        
        return self._get_obs(), reward, done, {}
    
    def render(self):
        """Print environment state (for debugging)"""
        print(f"Time Left: {self.minutes_left} mins | Odds: {self.odds} | Budget: {self.budget} | \
              Bets Remaining: {self.bets_remaining} | Exp Returns: (A: {self.expected_return_A}, Draw: {self.expected_return_Draw}, B: {self.expected_return_B})")

# Example usage
if __name__ == "__main__":
    env = SoccerBetEnv()
    obs = env.reset()

    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        obs, reward, done, _ = env.step(action)
        env.render()
