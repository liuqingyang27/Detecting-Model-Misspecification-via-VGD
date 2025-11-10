import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import jax.random as random
from functools import partial
from tqdm.auto import trange
from collections import namedtuple
from KGD import KGD
import matplotlib.pyplot as plt
from calculate_mmd import calculate_mmd_squared


class VGD:

    def __init__(self, log_prior, log_likelihood, data, kernel=None):

        self.log_prior = log_prior
        self.log_likelihood = log_likelihood
        self.data = data

        self.prior_score = grad(log_prior)
        self.likelihood_score = grad(log_likelihood, argnums=0)

        if kernel is None:
            self.kernel = self.rbf_kernel
        else:
            self.kernel = kernel

    def rbf_kernel(self, x, y, lengthscale=1):
        sq_dist = jnp.sum((x - y) ** 2)
        return jnp.exp(- sq_dist / (lengthscale ** 2))
    
    @staticmethod
    @partial(jit, static_argnames=('prior_score', 'likelihood_score', 'log_likelihood', 'kernel'))
    def _static_update(prior_score, likelihood_score, log_likelihood, data, kernel, particles, step_size, lengthscale=1):
        if particles.ndim == 1:
            particles = particles.reshape(-1, 1)  # Ensure particles are in the right shape
        n, d = particles.shape
        # Compute scores and probabilities
        prior_scores = vmap(prior_score)(particles)
        # likelihood_scores = vmap(lambda theta: jnp.sum(vmap(lambda x, y: likelihood_score(theta, x, y))(*data)))(particles)

        weighted_scores = vmap(lambda theta: 
                               jnp.sum(vmap(lambda x, y: 
                                    n * jnp.exp(log_likelihood(theta, x, y) - jax.scipy.special.logsumexp(vmap(lambda theta: log_likelihood(theta, x, y))(particles))) * likelihood_score(theta, x, y)
                                    )(*data), axis=0)
                               )(particles)
        weighted_scores = weighted_scores.reshape(-1, d)
        
        # Compute kernel and its gradient
        kernel_l = partial(kernel, lengthscale=lengthscale)
        kernel_matrix = vmap(lambda x: vmap(lambda y: kernel_l(x, y))(particles))(particles)
        d1_kernel = grad(kernel_l, argnums=0)
        d1_kernel_matrix = vmap(lambda x: vmap(lambda y: d1_kernel(x, y))(particles))(particles)

        # Compute weights
        prior_scores = prior_scores.reshape(-1, d)
        total_scores = prior_scores + weighted_scores
        # total_scores_SVGD = prior_scores + likelihood_scores.reshape(-1, d)
        
        
        term1 = jnp.sum(d1_kernel_matrix, axis=0)
        term1 = term1.reshape(-1, d)

        term2 = kernel_matrix @ total_scores
        

        # Update particles
        phi = (term1 + term2)/n 
        particles_VGD = particles + step_size * phi
        # phi_VGD = phi
        
        return particles_VGD, KGD(kernel).square_kgd(particles, S_PQ=total_scores)
    
    @staticmethod
    @partial(jit, static_argnames=('prior_score', 'likelihood_score', 'kernel'))
    def _static_SVGD_update(prior_score, likelihood_score, data, kernel, particles, step_size, lengthscale=1, indices=None):
        if indices is not None:
            data = (data[0][indices], data[1][indices])
        if particles.ndim == 1:
            particles = particles.reshape(-1, 1)  # Ensure particles are in the right shape
        n, d = particles.shape
        particles = particles.reshape(-1, d)  # Ensure particles are in the right shape

        # Compute kernel and its gradient
        kernel_l = partial(kernel, lengthscale=lengthscale)
        kernel_matrix = vmap(lambda x: vmap(lambda y: kernel_l(x, y))(particles))(particles)
        d1_kernel = grad(kernel_l, argnums=0)
        d1_kernel_matrix = vmap(lambda x: vmap(lambda y: d1_kernel(x, y))(particles))(particles)

        total_scores_SVGD = vmap(lambda theta: prior_score(theta) + likelihood_score(theta, *data))(particles).reshape(-1, d)

        term1 = jnp.sum(d1_kernel_matrix, axis=0)
        term1 = term1.reshape(-1, d)
        term2_SVGD = kernel_matrix @ total_scores_SVGD
        phi_SVGD = (term1 + term2_SVGD)/n
        particles_SVGD = particles + step_size * phi_SVGD
        return particles_SVGD, KGD(kernel).square_kgd(particles, S_PQ=total_scores_SVGD)
    
    def run(self, initial_particles, num_iterations, step_size, lengthscale=1):
        """
        Runs the VGD update for a specified number of iterations.
        
        Args:
            initial_particles (np.ndarray): Initial particles to start the update.
            num_iterations (int): Number of iterations to run the update.
            step_size (float): Step size for the update.
            lengthscale (float): Lengthscale for the RBF kernel.
        
        Returns:
            np.ndarray: Updated particles after all iterations.
        """
        jit_update = partial(self._static_update,
                             self.prior_score, self.likelihood_score, self.log_likelihood, self.data, self.kernel)
        jit_SVGD_update = partial(self._static_SVGD_update,
                                  self.prior_score, self.likelihood_score, self.data, self.kernel)
        self.particles = initial_particles.copy()
        self.particles_SVGD = initial_particles.copy()
        self.history = [initial_particles.copy()]
        self.history_SVGD = [initial_particles.copy()]
        self.KGD_history = []
        self.KSD_history = []

        for _ in trange(num_iterations):
            self.particles, current_KGD = jit_update(self.particles, step_size, lengthscale)
            self.particles_SVGD, current_KSD = jit_SVGD_update(self.particles_SVGD, step_size, lengthscale)
            self.history.append(self.particles.copy())
            self.history_SVGD.append(self.particles_SVGD.copy())
            self.KGD_history.append(current_KGD)
            self.KSD_history.append(current_KSD)

        self.particles = jnp.array(self.particles)
        self.history = jnp.array(self.history)
        self.particles_SVGD = jnp.array(self.particles_SVGD)
        self.history_SVGD = jnp.array(self.history_SVGD)
        self.KGD_history = jnp.array(self.KGD_history)
        self.KSD_history = jnp.array(self.KSD_history)

        return self.particles, self.history, self.particles_SVGD, self.history_SVGD, self.KGD_history, self.KSD_history


    # def lengthscale(self):
    #     all_particles = jnp.concatenate([self.particles, self.particles_SVGD], axis=0)
    #     vmapped_model = jax.vmap(self.model, in_axes=(0, None))
    #     all_results = vmapped_model(all_particles, self.x)
    #     self.mmd_length_scale = jnp.std(all_results)
    #     return self.mmd_length_scale

    # def mmd_squared(self, p=1, lengthscale=None):
    #     if lengthscale is None:
    #         lengthscale = self.lengthscale()
    #     return calculate_mmd_squared(self.particles_SVGD, self.particles, self.x, self.model, lengthscale, self.sigma, p)
