import numpy as np
from matplotlib import pyplot as plt
from scipy import stats

# containers
T = 500
xs = np.zeros(T+1)
ys = np.zeros(T)

# parameters
mu = 0.2
sigmav = 0.2
phi = 0.9

# initialization
x0 = 0
xs[0] = mu + np.random.randn()*sigmav/np.sqrt(1-phi**2)
ys[0]= (np.exp(xs[0])/2)*np.random.randn()

# simulate from model
for t in np.arange(1, T):
    xs[t] = mu + phi*(xs[t-1]-mu) + sigmav*np.random.randn()
    ys[t] = (np.exp(xs[t])/2)*np.random.randn()

# plot simulated data
plt.figure(figsize=(9,7))
fix, ax = plt.subplots(2,1)
ax[0].plot(xs); ax[1].plot(ys) ; plt.show()

class ParticleFilter(object):

    def __init__(self, nparticles):
        self.ys = None
        self.nparticles = nparticles


    def __init_containers(self):
        self.particles = np.zeros(shape=(self.nparticles, self.T))
        self.weights = np.ones(shape=(self.nparticles, self.T))
        self.norm_weights = np.zeros(shape=(self.nparticles, self.T))
        self.ancestors = np.zeros(shape=(self.nparticles, self.T))
        self.filtered_dist = np.zeros(shape=(self.T,1))
        self.loglike = 0

    def initialize(self, y, params):

        self.ys = y
        self.T = len(y)
        self.__init_containers()
        self.particles[:,0] = x0
        self.weights[:,0] = 1
        self.norm_weights[:,0] = 1/self.nparticles
        self.filtered_dist[0] = x0
        self.ancestors[:,0] = np.arange(self.nparticles)
        self.params = params

    def sample_ancestors(self, t):
        new_ancestors = np.random.choice(self.nparticles, self.nparticles, p=self.norm_weights[:,t-1], replace=True)
        #resample full lineage up until point t
        self.ancestors[:, 1:t-1] = self.ancestors[new_ancestors, 1:t-1] #reshuffle to sample lineages
        # then set current current anceestors at point t
        self.ancestors[:,t] = new_ancestors
        return new_ancestors

    def propagate_particles(self, t, new_ancestors):
        self.particles[:, t] = self.params.get('mu') + self.params.get('phi')*\
        ( self.particles[new_ancestors, t - 1] - self.params.get('mu') ) +\
        self.params.get('sigmav') * np.random.randn(1, self.nparticles)


    def compute_importance_weights(self, t):
        #compute weights using the observation equation
        ws = stats.norm.logpdf(self.ys[t-1], loc=0, scale=np.exp(self.particles[:,t]/2))
        #log shift the weights
        self.weights[:,t] = np.exp(ws - np.max(ws))
        self.norm_weights[:,t] = self.weights[:,t]  / np.sum(self.weights[:,t])
        #now update likelihood and filtering distribution
        predLL = np.max(ws) + np.log(np.sum(self.weights[:,t])) - np.log(self.nparticles)
        self.loglike += predLL
        self.filtered_dist[t] = np.sum( self.weights[:,t] * self.particles[:,t] ) / np.sum(self.weights[:,t])

    def run_filter(self, y, params):
        self.initialize(y, params)
        for t in np.arange(1, self.T):
            ancestor_idx = self.sample_ancestors(t)
            self.propagate_particles(t, ancestor_idx)
            self.compute_importance_weights(t)

pf = ParticleFilter(nparticles=750)

pf.run_filter(ys, {'mu': 0.4, 'phi':.8, 'sigmav':0.4})

plt.figure(figsize=(7,5)); plt.grid(alpha=0.25);
plt.plot(ys); plt.plot(pf.filtered_dist); plt.title("Signal vs. Filtered Distribution\n logL: %0.3f" % pf.loglike);
plt.show()


# PMMH Algorithm for parameter estimation

K=7500 #iterations

#set up containers
lls = np.zeros(K)
phis = np.zeros((K,3))
accepts = np.zeros(K)
phi_proposed= np.zeros((K,3))
ll_proposed = np.zeros(K)

#initialize chains
phi0 = [0.2, 0.85, 0.17]
phis[0,:] = phi0

pf.run_filter(ys, {'mu': phis[0,0],'phi':phis[1,0], 'sigmav':phis[2,0]})

lls[0] = pf.loglike
# set MC sampling parameters
step_size=(0.09**2, 0.09, 0.09)
# print log header
print("\niter\tmu\tphi\tsigma_v")
# run MCMC sampler
for k in np.arange(1, K):
    # Proposal Step
    phi_proposed[k,:] = phis[k-1,:] + np.random.randn(3)*step_size
    #conpute likelihood IF new parameter set is within bounds
    if (np.abs(phi_proposed[k,1]) < 1 & (phi_proposed[k,2] > 0)):
        pf.run_filter(ys, {'mu': phi_proposed[k,0], 'phi':phi_proposed[k,0], 'sigmav':phi_proposed[k,0]})
        ll_proposed[k] = pf.loglike
    # Metropolis Hastings Step
    #priors
    prior = stats.norm.logpdf(phi_proposed[k,0], 0, 0.2) - stats.norm.logpdf(phis[k-1,0], 0, 0.2) #mu prior
    #phi prior
    prior += stats.norm.logpdf(phi_proposed[k,1], 0.85,0.05) - stats.norm.logpdf(phis[k-1,1], 0.85, 0.05) #phi
    #sigmav prior
    prior += stats.gamma.logpdf(phi_proposed[k,2], 2.5, scale=1/10) - stats.gamma.logpdf(phis[k-1,2], 2.5, scale=1/10)
    #likelihood
    ll_diff = ll_proposed[k] - lls[k-1]

    a = np.min((1., np.exp(prior + ll_diff)))
    a = a * (np.abs(phi_proposed[k,1])<1 & (phi_proposed[k,2] > 0))
    u = np.random.uniform()

    if u < a:
        phis[k,:] = phi_proposed[k,:]
        lls[k] = ll_proposed[k]
        accepts[k] = 1
    else:
        phis[k,:] = phis[k-1,:]
        lls[k] = lls[k-1]
        accepts[k] = 0

    if k % 100 == 0:
        print("%d\t%0.3f\t%0.3f\t%0.3f" % (k, phis[k,0], phis[k,1], phis[k,2]))


plt.hist(phis[2500:,0])
