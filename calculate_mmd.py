import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
from functools import partial
from typing import Callable, Union


@partial(jax.jit, static_argnames=['fn', 'p'])
def kappa(theta_1: jnp.ndarray, 
          theta_2: jnp.ndarray, 
          x: Union[float, jnp.ndarray], 
          fn: Callable, 
          length_scale: float, 
          sigma: float, 
          p: int = 1) -> jnp.ndarray:
    const = jnp.sqrt(length_scale**2 / (length_scale**2 + 2 * sigma**2))**p

    fn_diff = fn(theta_1, x) - fn(theta_2, x)
    sq_dist = fn_diff * fn_diff
    
    exp_term = jnp.exp(-sq_dist / (2 * (length_scale**2 + 2 * sigma**2)))
    return const * exp_term

@partial(jax.jit, static_argnames=['fn', 'p'])
def calculate_mmd_squared(
    particles_SVGD: jnp.ndarray, 
    particles_VGD: jnp.ndarray, 
    x: Union[float, jnp.ndarray], 
    fn: Callable, 
    length_scale: float, 
    sigma: float, 
    p: int = 1) -> jnp.ndarray:
   
    N = particles_SVGD.shape[0]
    if N != particles_VGD.shape[0]:
        raise ValueError("dimensions of particles_SVGD and particles_VGD must be the same")

    def kappa_three_terms(theta_1, theta_2, x, fn, length_scale, sigma, p):
        return kappa(theta_1, theta_1, x, fn, length_scale, sigma, p) - 2 * kappa(theta_1, theta_2, x, fn, length_scale, sigma, p) + kappa(theta_2, theta_2, x, fn, length_scale, sigma, p)
        
    ## kappa matrix computation function
    k_vmapped = jax.vmap(
        jax.vmap(kappa_three_terms, in_axes=(None, 0, None, None, None, None, None)),
        in_axes=(0, None, None, None, None, None, None)
    )
    
    ## Compute the three terms of the MMD^2
    # term1_matrix = k_vmapped(particles_SVGD, particles_SVGD, x, fn, length_scale, sigma, p)
    # term1 = jnp.sum(term1_matrix, axis=(0, 1))

    # term2_matrix = k_vmapped(particles_SVGD, particles_VGD, x, fn, length_scale, sigma, p)
    # term2 = jnp.sum(term2_matrix, axis=(0, 1))

    # term3_matrix = k_vmapped(particles_VGD, particles_VGD, x, fn, length_scale, sigma, p)
    # term3 = jnp.sum(term3_matrix, axis=(0, 1))

    terms_matrix = k_vmapped(particles_SVGD, particles_VGD, x, fn, length_scale, sigma, p)

    # mmd2 = jnp.mean((term1 - 2 * term2 + term3) / (N**2))
    mmd2 = jnp.mean(terms_matrix)/(N**2)
    return mmd2
