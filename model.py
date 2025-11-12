import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import jax.random as random

class model:
    def __init__(self, sigma, fn, theta_dim):
        ## y=fn(theta, x) + N(0, I_dim*sigma^2)
        self.fn = fn
        self.sigma = sigma
        self.dim = theta_dim

    def log_prior(self, theta):
        return jax.scipy.stats.norm.logpdf(theta, loc=jnp.zeros_like(theta), scale=1.0).sum()
    
    def log_likelihood(self, theta, x, y):
        predicted_y = self.fn(theta, x)
        log_probs = jax.scipy.stats.norm.logpdf(y, loc=predicted_y, scale=self.sigma)
        return log_probs.sum()
    
    def generate_x(self, n_data, x_min=-1.0, x_max=1.0, key=random.PRNGKey(0)):
        x = random.uniform(key, (n_data,), minval=x_min, maxval=x_max)
        return x

    def generate_data(self, n_data, particle, x_min=-1.0, x_max=1.0, key=random.PRNGKey(0)):
        subkey1, subkey2 = random.split(key)
        x = self.generate_x(n_data, x_min, x_max, key=subkey1)

        particle_jnp = jnp.asarray(particle)
        particle_arr = jnp.atleast_1d(particle_jnp)
        y = self.fn(particle_arr, x)
        noise = self.sigma * jax.random.normal(subkey2, shape=y.shape)
        y_noised = y + noise
        return (x, y_noised)
    