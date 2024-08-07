import  numpy as np
import matplotlib.pyplot as plt


NUM_ITERS = 100000
# EPS = 0.1
BANDITS_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
	def __init__(self, p):
		self.win_rate = p
		self.win_estimate = 0.
		self.num_wins = 0
		# This must be intialized with 1
		self.num_plays = 0
		# self.p_estimate_list = [self.win_estimate]
		self.p_estimate_list = [self.win_estimate]

	def play(self):
		stat = np.random.random() < self.win_rate
		self.num_plays += 1
		if stat:
			self.num_wins += 1
		return stat

	def update_estimate(self, last_try):
		self.win_estimate = ((self.num_plays - 1) * self.win_estimate + last_try) / (self.num_plays)

	def update_p_estimate_list(self):
		self.p_estimate_list.append(self.win_estimate)


def ucb(mean, n, n_j):
	return mean + np.sqrt(2 * np.log(n) / n_j)

def experiment():
	bandits = [Bandit(band_prob) for band_prob in BANDITS_PROBABILITIES]
	rewards = np.zeros(NUM_ITERS)
	total_plays = 0
	for j in range(len(bandits)):
		win_or_not = bandits[j].play()
		total_plays += 1
		bandits[j].update_estimate(win_or_not)
		bandits[j].update_p_estimate_list()

	num_times_explored = 0
	num_times_exploited = 0
	optimal_bandit_palyed = 0
	
	optimal_bandit_idx = np.argmax([bandit.win_rate for bandit in bandits])

	for i in range(NUM_ITERS):
		j = np.argmax([ucb(bandit.win_estimate, total_plays, bandit.num_plays) for bandit in bandits])

		win_or_not = bandits[j].play()
		rewards[i] = win_or_not

		total_plays += 1
		bandits[j].update_estimate(win_or_not)
		bandits[j].update_p_estimate_list()

	print("Best Bandit: ", np.argmax([bandit.win_estimate for bandit in bandits]))
	cumulative_reward = np.cumsum(rewards) / np.arange(1, NUM_ITERS + 1)

	if j == optimal_bandit_idx:
		optimal_bandit_palyed += 1
	print(f"Optimal bandit {np.argmax(BANDITS_PROBABILITIES)}")
	print(f"total rewards: {rewards.sum() / NUM_ITERS}")
	print(f"Number of times explored {num_times_explored}")
	print(f"Number of times exploited {num_times_exploited}")
	print("############################################################")
	for bandit in bandits:
		print("Mean win estimate of the bandit:", bandit.win_estimate)
		print(f"This bandit was played {bandit.num_plays} times")
		print(f"This bandit won {bandit.num_wins} times")
		print("Actual win rate of the bandit:", bandit.win_rate)
		print("average reward of the bandit:", bandit.num_wins / bandit.num_plays)
		print("############################################################")
		plt.plot(np.array(bandit.p_estimate_list))
		plt.plot(np.ones(NUM_ITERS) * bandit.win_rate)
		plt.xscale("log")
		plt.show()

	plt.plot(np.ones(NUM_ITERS) * np.max(BANDITS_PROBABILITIES))
	plt.plot(cumulative_reward)
	plt.title("cumulative sum of rewards")
	plt.xscale("log")
	plt.show()


if __name__ == "__main__":
	experiment()