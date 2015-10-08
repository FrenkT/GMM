import numpy as np
from GMM import GMM

if __name__ == '__main__':
    group_a = np.random.normal(loc=(20.00, 14.00), scale=(4.0, 4.0), size=(1000, 2))
    group_b = np.random.normal(loc=(15.00, 8.00), scale=(2.0, 2.0), size=(1000, 2))

    data = np.concatenate((group_a, group_b))

    g = GMM()
    g.train(data)
    for c in g.components:
        print '*****'
        print c.mean
        print c.cov
