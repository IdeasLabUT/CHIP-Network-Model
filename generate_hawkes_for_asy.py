import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import generative_model_utils as utils


plot_path = '/shared/Results/CommunityHawkes/plots/'


def calc_event_count_man_var(mu, alpha, beta, t, n_itter):
    event_counts = np.zeros(n_itter, dtype=np.int)
    for i in range(n_itter):
        event_counts[i] = len(utils.simulate_univariate_hawkes(mu, alpha, beta, t))

    asy_mean = utils.asymptotic_mean(mu, alpha, beta, t)
    sam_mean = np.mean(event_counts)

    asy_var = utils.asymptotic_var(mu, alpha, beta, t)
    sam_var = np.var(event_counts)
    print(t)
    return asy_mean, sam_mean, asy_var, sam_var, t


n_itter = 100000
# n_itter = 1

mu = 0.5
alpha = 0.6
beta = 0.8

t = 10

t_values = [1, 2, 5, 10, 20, 35, 50, 60, 85, 100, 125, 150, 175, 200,
            225, 250, 275, 300, 325, 350, 375, 400, 425, 450, 475, 500, 600, 800, 1000]

num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(calc_event_count_man_var)(mu, alpha, beta, t, n_itter) for t in t_values)
# each row is asy_mean, sam_mean, asy_var, sam_var, t
results = np.asarray(results, dtype=np.float)

# n_t_vals = len(t_values)
# asy_means = np.zeros(n_t_vals, dtype=np.float)
# sam_means = np.zeros(n_t_vals, dtype=np.float)
# asy_vars = np.zeros(n_t_vals, dtype=np.float)
# sam_vars = np.zeros(n_t_vals, dtype=np.float)
# for idx, t in enumerate(t_values):
#     calc_event_count_man_var(mu, alpha, beta, t, n_itter)
#     asy_means[idx] = utils.asymptotic_mean(mu, alpha, beta, t)
#     sam_means[idx] = np.mean(event_counts)
#
#     asy_vars[idx] = utils.asymptotic_var(mu, alpha, beta, t)
#     sam_vars[idx] = np.var(event_counts)
#
#     # print(f"Asy Mean: {asy_means[idx]:.3f}")
#     # print(f"Sample Mean: {sam_means[idx]:.3f}")
#     # print(f"Variance: {asy_vars[idx]:.3f}")
#     # print(f"Variance: {sam_vars[idx]:.3f}")
#     print(t)

# plt.plot(results[:, 4], results[:, 0], label="Asymptotic", linestyle='-', marker='o')
# plt.plot(results[:, 4], results[:, 1], label='sample', linestyle='-', marker="*")
# plt.title("Hawkes Generated Event Counts Asymptotic vs. Sample Mean")
# plt.ylabel("Event Count Mean")
# plt.xlabel("T")
# plt.legend()
# plt.savefig(plot_path + "asy-mean.pdf")
# # plt.show()
#
# plt.clf()
#
# plt.plot(results[:, 4], results[:, 2], label="Asymptotic", linestyle='-', marker='o')
# plt.plot(results[:, 4], results[:, 3], label='sample', linestyle='-', marker="*")
# plt.title("Hawkes Generated Event Counts Asymptotic vs. Sample Variance")
# plt.ylabel("Event Count Variance")
# plt.xlabel("T")
# plt.legend()
# plt.savefig(plot_path + "asy-var.pdf")
# # plt.show()
#
# plt.clf()

plt.plot(results[:, 4], 100 * (results[:, 0] - results[:, 1]) / results[:, 0], linestyle='-', marker='o')
plt.title("Hawkes Generated Event Counts \n Asymptotic vs. Sample Mean Percent Difference \n (Asy - Samp) / Asy")
plt.ylabel("Event Count Mean")
plt.xlabel("T")
# plt.xticks(results[:, 4], results[:, 4])
plt.savefig(plot_path + "asy-mean-pd.pdf")
# plt.show()

plt.clf()

plt.plot(results[:, 4], 100 * (results[:, 2] - results[:, 3]) / results[:, 2], linestyle='-', marker='o')
plt.title("Hawkes Generated Event Counts \n Asymptotic vs. Sample Variance Percent Difference \n (Asy - Samp) / Asy")
plt.ylabel("Event Count Variance")
plt.xlabel("T")
# plt.xticks(results[:, 4], results[:, 4])
plt.savefig(plot_path + "asy-var-pd.pdf")
# plt.show()