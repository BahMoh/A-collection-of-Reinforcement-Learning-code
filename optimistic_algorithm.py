import numpy as np 
import matplotlib.pyplot as plt



NUM_ITERS = 1000
EPS = 0.1
BANDITS_PROBABILITIES = [0.2, 0.5, 0.75]




class Bandit:
	def __init__(self, p):
		self.win_rate = p
		self.win_estimate = 10

		self.num_wins = 0
		# This must be intialized with 1
		self.num_plays = 1
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

def experiment(EPS):
	bandits = [Bandit(band_prob) for band_prob in BANDITS_PROBABILITIES]

	# We initialize the initial estimates as below
	# infact we are trying this in a way to see how powerful this optimistic algorithm is 
	# The least probable bandit has the biggest initail estiamte
	# bandits[0].win_estimate = 30
	# bandits[1].win_estimate = 20
	# bandits[2].win_estimate = 10

	# Another thing to consider is we can assign small value to the most probable bandit

	bandits[0].win_estimate = 30
	bandits[1].win_estimate = 20
	bandits[2].win_estimate = 0.2
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
		bandits[j].update_p_estimate_list()
		# EPS = 0.999 * EPS
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

	print("Best slot machine:", np.argmax([bandit.win_estimate for bandit in bandits]))
	cumulative_reward = np.cumsum(rewards) / np.arange(1, NUM_ITERS + 1)

	plt.plot(np.ones(NUM_ITERS) * np.max(BANDITS_PROBABILITIES))
	plt.plot(cumulative_reward)
	plt.title("cumulative sum of rewards")
	plt.xscale("log")
	plt.show()


if __name__ == "__main__":
	experiment(EPS)