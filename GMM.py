import numpy as np
import matplotlib.pyplot as plt

from utils import compute_pdf


group_a = np.random.normal(loc=(20.00, 14.00), scale=(4.0, 4.0), size=(500, 2))
group_b = np.random.normal(loc=(15.00, 8.00), scale=(2.0, 2.0), size=(500, 2))

# plt.plot(group_a[:, 0], group_a[:, 1], 'ro')
# plt.plot(group_b[:, 0], group_b[:, 1], 'bo')
# plt.show()

data = np.concatenate((group_a, group_b))
mean_ext = np.mean(data, axis=0)
std_ext = np.std(data, axis=0)
# print std_ext
# plt.plot(data[:, 0], data[:, 1], 'ro')
# plt.plot(mean_ext[0], mean_ext[1], 'bo')

distr = []
distr.append({'weight': 0.5, 'mean': (0, 0), 'var': (0, 0), 'xi': 0, 'lam': 1, 'sig': 0, 'ni': 0})
distr.append({'weight': 0.5, 'mean': (0, 0), 'var': (0, 0), 'xi': 0, 'lam': 1, 'sig': 0, 'ni': 0})

# split centroid
distr[0]['mean'] = mean_ext + np.multiply(std_ext, 0.1)
distr[1]['mean'] = mean_ext - np.multiply(std_ext, 0.1)
# distr[0]['var'] = (4.0, 4.0)
# distr[1]['var'] = (2.0, 2.0)
distr[0]['var'] = np.eye(2)*(std_ext**2)
distr[1]['var'] = np.eye(2)*(std_ext**2)
distr[0]['ni'] = distr[0]['weight'] * len(data)
distr[1]['ni'] = distr[1]['weight'] * len(data)
# plt.plot(distr[0]['mean'][0], distr[0]['mean'][1], 'go')
# plt.plot(distr[1]['mean'][0], distr[1]['mean'][1], 'go')
# plt.show()

print ' distr 1'
print distr[0]['mean']
print distr[0]['var']

print ' distr 2'
print distr[1]['mean']
print distr[1]['var']
eval_train = []

for iter in range(20):
    print iter

    gamma_denoms = []
    for x in data:
        gamma_denom = 0
        for d in distr:
            gamma_denom += np.multiply(d['weight'], compute_pdf(x, d['mean'], d['var']))
        gamma_denoms.append(gamma_denom)
    sum_lik = sum(np.log(gamma_denoms))
    eval_train.append(sum_lik)

    gammas = []
    for x, i in zip(data, range(len(data))):
        num = np.multiply(distr[0]['weight'], compute_pdf(x, distr[0]['mean'], distr[0]['var']))
        gamma = np.divide(num, gamma_denoms[i])
        gammas.append(gamma)
    distr[0]['xi'] = sum(gammas)
    mi_xk = np.dot(gammas, data)
    mi_xk = np.divide(mi_xk, distr[0]['xi'])
    new_mean_1 = mi_xk
    sigma_xk = 0
    for i in range(len(data)):
        sigma_xk += np.multiply(gammas[i], np.dot(np.transpose(np.matrix(data[i]-mi_xk)), np.matrix(data[i]-mi_xk)))
    sigma_xk = np.divide(sigma_xk, distr[0]['xi'])
    new_var_1 = sigma_xk

    gammas = []
    for x, i in zip(data, range(len(data))):
        num = np.multiply(distr[1]['weight'], compute_pdf(x, distr[1]['mean'], distr[1]['var']))
        gamma = np.divide(num, gamma_denoms[i])
        gammas.append(gamma)
    distr[1]['xi'] = sum(gammas)
    mi_xk = np.dot(gammas, data)
    mi_xk = np.divide(mi_xk, distr[1]['xi'])
    new_mean_2 = mi_xk
    sigma_xk = 0
    for i in range(len(data)):
        sigma_xk += np.multiply(gammas[i], np.dot(np.transpose(np.matrix(data[i]-mi_xk)), np.matrix(data[i]-mi_xk)))
    sigma_xk = np.divide(sigma_xk, distr[1]['xi'])
    new_var_2 = sigma_xk

    denom_weight = 0
    for d in distr:
        denom_weight += d['xi']

    new_weight_1 = np.divide((distr[0]['xi']), denom_weight)
    new_weight_2 = np.divide((distr[1]['xi']), denom_weight)

    distr[0]['mean'] = new_mean_1
    distr[0]['var'] = new_var_1
    distr[0]['weight'] = new_weight_1
    distr[1]['mean'] = new_mean_2
    distr[1]['var'] = new_var_2
    distr[1]['weight'] = new_weight_2
    print new_weight_1, new_weight_2


print distr[0]['mean']
print np.sqrt(distr[0]['var'])
print distr[1]['mean']
print np.sqrt(distr[1]['var'])

plt.plot(eval_train)
plt.show()
