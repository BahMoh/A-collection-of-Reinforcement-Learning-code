import matplotlib.pyplot as plt
import numpy as np

NUM_ITERS = 100000
EPS = 0.1 
BANDITS_PROBABILITIES = [0.2, 0.4, 0.5, 0.7]


class SlotMachine:
	def __init__(self, win_rate):
		self.win_rate = win_rate
		self.win_estimate = 0.0

		self.num_wins = 0
		self.num_plays = 0


	def play(self):
		stat = np.random.random() < self.win_rate
		self.num_plays += 1
		if stat:
			self.num_wins += 1
		return stat

	def update_estimate(self, last_try):
		self.win_estimate = ((self.num_plays - 1) * self.win_estimate + last_try) / (self.num_plays)


def experiment(EPS):
	bandits = [SlotMachine(band_prob) for band_prob in BANDITS_PROBABILITIES]
	rewards = np.zeros(NUM_ITERS)

	num_times_explored = 0
	num_times_exploited = 0

	optimal_badit_palyed = 0
	
	optimal_bandit_idx = np.argmax([bandit.win_rate for bandit in bandits])
	for i in range(NUM_ITERS):
		if np.random.random() < EPS:
			num_times_explored += 1
			j = np.random.randint(0, len(bandits))
		else:
			num_times_exploited += 1
			j = np.argmax([bandit.win_estimate for bandit in bandits])

		if j == optimal_bandit_idx:
			optimal_badit_palyed += 1


		win_or_not = bandits[j].play()
		rewards[i] = win_or_not
		bandits[j].update_estimate(win_or_not)
		# EPS = 0.999 * EPS

	print(f"total rewards: {rewards.sum() / NUM_ITERS}")
	print(f"Number of times explored {num_times_explored}")
	print(f"Number of times exploited {num_times_exploited}")
	for bandit in bandits:
		print("Mean win estimate of the bandit:", bandit.win_estimate)
		print(f"This bandit was played {bandit.num_plays} times")
		print(f"This bandit won {bandit.num_wins} times")
		print("Actual win rate of the bandit:", bandit.win_rate)
		print("average reward of the bandit:", bandit.num_wins / bandit.num_plays)
		print("############################################################")

	print("Best slot machine:", np.argmax([bandit.win_estimate for bandit in bandits]))
	cumulative_reward = np.cumsum(rewards) / np.arange(1, NUM_ITERS + 1)

	plt.plot(np.ones(NUM_ITERS) * np.max(BANDITS_PROBABILITIES))
	plt.plot(cumulative_reward)
	plt.xscale("log")
	plt.show()


if __name__ == "__main__":
	experiment(EPS)