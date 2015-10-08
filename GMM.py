import numpy as np
import matplotlib.pyplot as plt

from utils import compute_pdf


group_a = np.random.normal(loc=(20.00, 14.00), scale=(4.0, 4.0), size=(1000, 2))
group_b = np.random.normal(loc=(15.00, 8.00), scale=(2.0, 2.0), size=(1000, 2))

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
distr.append({'weight': 0.5, 'mean': (0, 0), 'var': (0, 0), 'xi': 0, 'lam': 1, 'sig': 0, 'ni': 0, 'wpdf': [], 'gammas': 0})
distr.append({'weight': 0.5, 'mean': (0, 0), 'var': (0, 0), 'xi': 0, 'lam': 1, 'sig': 0, 'ni': 0, 'wpdf': [], 'gammas': 0})

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

    for d in distr:
        d['wpdf'] = []
    for x in data:
        for d in distr:
            d['wpdf'].append(np.multiply(d['weight'], compute_pdf(x, d['mean'], d['var'])))

    gamma_denoms = np.zeros(len(data))
    for d in distr:
        gamma_denoms += d['wpdf']
    sum_lik = sum(np.log(gamma_denoms))
    eval_train.append(sum_lik)

    for d in distr:
        d['gammas'] = np.divide(d['wpdf'], gamma_denoms)

    for d, i in zip(distr, range(len(distr))):
        d['xi'] = sum(d['gammas'])
        mi_xk = np.dot(d['gammas'], data)
        mi_xk = np.divide(mi_xk, d['xi'])
        sigma_xk = np.dot(np.dot(np.transpose(data-mi_xk), np.diag(d['gammas'])), (data-mi_xk))
        sigma_xk = np.divide(sigma_xk, d['xi'])
        d['mean'] = mi_xk
        d['var'] = sigma_xk

    denom_weight = 0
    for d in distr:
        denom_weight += d['xi']
    for d in distr:
        d['weight'] = np.divide(d['xi'], denom_weight)


print distr[0]['mean']
print distr[0]['var']
print distr[1]['mean']
print distr[1]['var']

plt.plot(eval_train)
plt.show()
