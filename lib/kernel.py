import jax.numpy as jnp

def rbf_kernel(x, y, h):
    diff = x - y
    return jnp.exp(-jnp.dot(diff, diff) / (2 * h**2))

def imq_kernel(x, y, c=1.0, lengthscale=1.0, beta=0.5):
    """ Inverse multiquadric Kernel."""

    # Compute the squared distance between x and y
    sqdist = jnp.sum((x - y) ** 2)
    
    # Compute the inverse multiquadric kernel value
    kernel_value = (c ** 2 + sqdist / lengthscale ** 2) ** (-beta)

    return kernel_value