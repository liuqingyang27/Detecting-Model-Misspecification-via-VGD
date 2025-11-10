import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import jax.random as random

class model:
    def __init__(self, sigma, fn, theta_dim):
        ## y=fn(theta, x) + N(0, I_dim*sigma^2)
        self.fn = fn
        self.sigma = sigma
        self.key = random.PRNGKey(49)
        self.dim = theta_dim

    def log_prior(self, theta):
        return jax.scipy.stats.norm.logpdf(theta, loc=jnp.zeros_like(theta), scale=1.0).sum()
    
    def log_likelihood(self, theta, x, y):
        predicted_y = self.fn(theta, x)
        log_probs = jax.scipy.stats.norm.logpdf(y, loc=predicted_y, scale=self.sigma)
        return log_probs.sum()
    
    # def generate_particles(self, n_particles=20):
    #     self.key, subkey = random.split(self.key)
    #     self.initial_particles = random.normal(subkey, shape=(n_particles, self.dim))

    # def get_initial_particles(self):
    #     return self.initial_particles

    def generate_x(self, n_data, x_min=-1.0, x_max=1.0):
        self.key, subkey = random.split(self.key)
        self.x = random.uniform(subkey, (n_data,), minval=x_min, maxval=x_max)

    def generate_data_batch(self, n_data, particles):
        self.generate_x(n_data)
        particles_jnp = jnp.asarray(particles) 
        particles_arr = jnp.atleast_1d(particles_jnp)
        y = vmap(self.fn, in_axes=(0, None))(particles_arr, self.x)
        noise = self.sigma * jax.random.normal(self.key, shape=y.shape)
        return (y + noise)[0]
    
