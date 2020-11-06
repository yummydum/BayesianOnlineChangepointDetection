import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy import stats
np.random.seed(42)


class BOCD:
    def __init__(self, dist, hazard, threshold=0.3):
        self.dist = dist
        self.hazard = hazard
        self.threshold = threshold

        self.t = 0
        self.cp = set()

        self.P = np.zeros((50, 50))
        self.P[0, 0] = 1
        self.message = np.array([1])

    def update(self, x):

        self.t += 1
        pis = self.dist.ppd(x)  # len(pis) == self.t

        growth_probs = pis * self.message * (1 - self.hazard)
        cp_prob = sum(pis * self.message * self.hazard)

        new_joint = np.append(cp_prob, growth_probs)

        if self.t + 1 > len(self.P):
            self.expandP()
        self.P[self.t, :self.t + 1] = new_joint

        evidence = np.sum(new_joint)
        if evidence > 0:
            self.P[self.t, :self.t + 1] /= evidence

        self.dist.update_theta(x)
        self.message = new_joint

        # Decide change point
        max_p = self.P[self.t, :].max()
        if max_p >= self.threshold:
            cp = self.t - self.P[self.t, :].argmax()
            if cp > 0 and cp not in self.cp:
                self.cp.add(cp)
                print(f'Change point {cp} found at t={self.t}')
        return

    def expandP(self):
        N = len(self.P) + 50
        newP = np.zeros((N, N))
        r, c = self.P.shape
        newP[:r, :c] = self.P
        self.P = newP
        return

    def plot(self, data):
        fig, axes = plt.subplots(2, 1, figsize=(20, 10))
        ax1, ax2 = axes

        T = len(data)
        ax1.scatter(range(0, T), data)
        ax1.plot(range(0, T), data)
        ax1.set_xlim([0, T])
        ax1.margins(0)

        ax2.imshow(np.rot90(self.P),
                   aspect='auto',
                   cmap='gray_r',
                   norm=LogNorm(vmin=0.0001, vmax=1))
        ax2.set_xlim([0, T])
        ax2.margins(0)

        for x in self.get_cp():
            ax1.axvline(x, c='red', ls='dotted')
            ax2.axvline(x, c='red', ls='dotted')

        return ax1, ax2

    def dist_t(self, t):
        return self.P[t, :][self.P[t, :] > 0]

    def get_cp(self):
        return sorted([x for x in self.cp])


class Gaussian:
    def __init__(self, alpha=1, beta=1, kappa=1, mu=0):
        self.alpha0 = self.alpha = np.array([alpha])
        self.beta0 = self.beta = np.array([beta])
        self.kappa0 = self.kappa = np.array([kappa])
        self.mu0 = self.mu = np.array([mu])

    def ppd(self, data):
        """
        Note: Posterior predictive distribution for gaussian is student t
        """
        return stats.t.pdf(x=data,
                           df=2 * self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa + 1) /
                                         (self.alpha * self.kappa)))

    def update_theta(self, data):
        muT0 = np.concatenate(
            (self.mu0, (self.kappa * self.mu + data) / (self.kappa + 1)))
        kappaT0 = np.concatenate((self.kappa0, self.kappa + 1.))
        alphaT0 = np.concatenate((self.alpha0, self.alpha + 0.5))
        betaT0 = np.concatenate(
            (self.beta0, self.beta + (self.kappa * (data - self.mu)**2) /
             (2. * (self.kappa + 1.))))
        self.mu = muT0
        self.kappa = kappaT0
        self.alpha = alphaT0
        self.beta = betaT0


class Poisson:
    def __init__(self, shape=1, rate=1):
        self.shape0 = self.shape = np.array([shape])
        self.rate0 = self.rate = np.array([rate])

    def ppd(self, data):
        """
        Note: Posterior predictive distribution for poisson is negative binomial
        """
        return stats.nbinom.pmf(k=data, n=self.shape,
                                p=1 / (1 + self.rate))  # rate = p/(1-p)

    def update_theta(self, data):
        shapeT0 = np.concatenate((self.shape0, self.shape + data))
        rateT0 = np.concatenate((self.rate0, self.rate / (self.rate + 1)))

        self.shape = shapeT0
        self.rate = rateT0