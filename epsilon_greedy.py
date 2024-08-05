import matplotlib.pyplot as plt
import numpy as np

# This is the code with a decaying epsilon

NUM_TRIALS = 10000
EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
	def __init__(self, win_prob):
		self.win_prob = win_prob
		self.p_estimate = 0.0
		# Number of samples
		self.N = 0

	def pull(self):
		return np.random.random() < self.win_prob

	def update(self, x):
		self.N += 1
		self.p_estimate = ((self.N - 1) * self.p_estimate + x) / self.N


def experiment(EPS):
	bandits = [Bandit(win_prob) for win_prob in BANDIT_PROBABILITIES]

	rewards = np.zeros(NUM_TRIALS)
	num_times_explored = 0
	num_times_exploited = 0
	num_optimal = 0
	optimal_j = np.argmax([b.win_prob for b in bandits])
	print("Optimal j:", optimal_j)


	for i in range(NUM_TRIALS):
		EPS = EPS * 0.999
		# use epsilon-greedy to select next bandit
		if np.random.random() < EPS:
			num_times_explored += 1
			# Is the index of the bandit
			j = np.random.randint(0,3)

		else:
			num_times_exploited += 1
			j = np.argmax([b.p_estimate for b in bandits])

		if j == optimal_j:
			num_optimal += 1

		# Pull the arm for the bandit with the largest sample
		x = bandits[j].pull()

		# Updata rewards log
		rewards[i] = x

		# Update the distribution for the bandit whose arm we just pulled
		bandits[j].update(x)


	for b in bandits:
		print("Mean estimate:", b.p_estimate)


		print(f"Total reward Earned: {rewards.sum()}")
		print(f"Overal win rate: {rewards.sum() / NUM_TRIALS}")
		print(f"Nume times explored: {num_times_explored}")
		print(f"Num times exploited: {num_times_exploited}")
		print(f"Num times selected optimal bandit: {num_optimal}")

		cumulative_rewards = np.cumsum(rewards)
		print(cumulative_rewards, "cumulative_rewards")
		win_rates = cumulative_rewards / np.arange(1, NUM_TRIALS + 1)
		# print(win_rates, " b")
		plt.plot(win_rates)
		plt.plot(np.ones(NUM_TRIALS) * np.max(BANDIT_PROBABILITIES))
		plt.xscale("log")
		plt.show()		

if __name__ == "__main__":
	experiment(EPS)
