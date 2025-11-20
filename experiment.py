import jax
import jax.numpy as jnp
from jax import lax
from jax import grad, vmap, jit
import jax.random as random
from functools import partial

import matplotlib.pyplot as plt


from model import model
from methods import VGD
from calculate_mmd import calculate_mmd_squared

class experiment:
    def __init__(self, model, data, n_particles=20, kernel=None, key=random.PRNGKey(49)):
        self.model = model
        self.fn = model.fn
        self.sigma = model.sigma
        self.log_prior = model.log_prior
        self.log_likelihood = model.log_likelihood
        self.dim = model.dim

        self.data = data
        self.key = key
        self.n_particles = n_particles
        self.kernel = kernel
        

        self.initialise_particles()
        self.algorithm = VGD(self.log_prior, self.log_likelihood, self.data, kernel=kernel)


    def initialise_particles(self):
        self.key, subkey = random.split(self.key)
        self.initial_particles = random.normal(subkey, shape=(self.n_particles, self.dim))
    
    def run(self, n_steps=1000, step_size=0.01, lengthscale=None):
        self.n_steps = n_steps
        self.step_size = step_size
        self.lengthscale = lengthscale
        self.particles_VGD, self.history_VGD, self.particles_SVGD, self.history_SVGD, self.history_KGD, self.history_KSD = self.algorithm.run(self.initial_particles, num_iterations=self.n_steps, step_size=self.step_size, lengthscale=self.lengthscale)

    def mmd_squared(self):
         # Compute MMD length scale and actual MMD
        all_particles = jnp.concatenate([self.particles_VGD, self.particles_SVGD], axis=0)
        vmapped_model = jax.vmap(self.fn, in_axes=(0, None))
        all_results = vmapped_model(all_particles, self.data[0])
        self.mmd_length_scale = jnp.std(all_results)
        # self.mmd_length_scale = jnp.ptp(all_results)
        self.actual_mmd = calculate_mmd_squared(self.particles_SVGD, self.particles_VGD, self.data[0], self.fn, self.mmd_length_scale, self.sigma, p=1)
        return self.actual_mmd

    def plot_KGD(self):
        plt.plot(range(len(self.history_KGD)), jnp.log(self.history_KGD))
        plt.xlabel('Iteration number')
        plt.ylabel('Log KGD')
        plt.title('Log KGD over Iterations')
        plt.show()

    def plot_KSD(self):
        plt.plot(range(len(self.history_KSD)), jnp.log(self.history_KSD))
        plt.xlabel('Iteration number')
        plt.ylabel('Log KSD')
        plt.title('Log KSD over Iterations')
        plt.show()

    def plot_both(self):
        plt.plot(range(len(self.history_KGD)), jnp.log(self.history_KGD), label='Log KGD', color='#ff7f0e')
        plt.plot(range(len(self.history_KSD)), jnp.log(self.history_KSD), label='Log KSD', color='#1f77b4')
        plt.xlabel('Iteration number')
        plt.ylabel('Log Value')
        plt.title('Log KGD and Log KSD over Iterations')
        plt.legend()
        plt.show()

class diagnostic_experiment(experiment):
    def __init__(self, experiment, key=random.PRNGKey(49)):
        super().__init__(experiment.model, experiment.data, experiment.n_particles, experiment.kernel, key=key)
        self.x, self.y = self.data
        self.particles_VGD, self.history_VGD, self.particles_SVGD, self.history_SVGD, self.history_KGD, self.history_KSD = experiment.particles_VGD, experiment.history_VGD, experiment.particles_SVGD, experiment.history_SVGD, experiment.history_KGD, experiment.history_KSD
        self.n_steps = experiment.n_steps
        self.step_size = experiment.step_size
        self.lengthscale = experiment.lengthscale

         # Compute MMD length scale and actual MMD
        all_particles = jnp.concatenate([self.particles_VGD, self.particles_SVGD], axis=0)
        vmapped_model = jax.vmap(self.fn, in_axes=(0, None))
        all_results = vmapped_model(all_particles, self.x)
        self.mmd_length_scale = jnp.std(all_results)
        # self.mmd_length_scale = jnp.ptp(all_results)
        self.actual_mmd = calculate_mmd_squared(self.particles_SVGD, self.particles_VGD, self.x, self.fn, self.mmd_length_scale, self.sigma, p=1)


    def sample_particles(self, particles, n, repeat=True, key=random.PRNGKey(0)):
        ## Sample n particles from Q_Bayes. We actually return a vector of posterior means, but sampling is also possible.
        if repeat:
            mean = jnp.mean(particles, axis=0)
            return jnp.tile(mean, (n, 1))
        num_particles = particles.shape[0]
        indices = random.choice(key, num_particles, shape=(n,), replace=False)
        return particles[indices]
    
    @staticmethod
    @partial(jit, static_argnums=(2,))
    def generate_data_batch(particles, x, fn, sigma, key): 
        ## Generate datasets for each theta in particles. {y_i,j} ~ p(y|x_j, theta_i)
        particles_jnp = jnp.asarray(particles)    
        particles_arr = jnp.atleast_1d(particles_jnp)
        y = vmap(fn, in_axes=(0, None))(particles_arr, x)
        noise = sigma * jax.random.normal(key, shape=y.shape)
        return y + noise

    def run_single_experiment(self, data_y, 
                                initial_particles_single):
        ## VGD and SVGD functions for vmap
        data_for_vgd = (self.x, data_y)

        algorithm = VGD(self.log_prior, self.log_likelihood, data_for_vgd, kernel=self.kernel)
        
        particles_VGD, history_VGD, particles_SVGD, history_SVGD, history_KGD, history_KSD = \
            algorithm.run(initial_particles_single, num_iterations=self.n_steps, 
                        step_size=self.step_size, lengthscale=self.lengthscale)
    
        return particles_VGD, history_VGD, particles_SVGD, history_SVGD, history_KGD, history_KSD
    
    def resample_experiment(
            self,
            num_sample_from_posterior,
            parallel=True,
            trajectory=True
            ):
        ## Resample theta and run corresponding experiments
        self.key, _ = random.split(self.key)
        initial_particles = random.normal(self.key, (self.n_particles, self.particles_SVGD.shape[1]))

        subkey1, subkey2 = random.split(self.key)
        new_particles = self.sample_particles(self.particles_SVGD, num_sample_from_posterior, key=subkey1)

        dataset = self.generate_data_batch(new_particles, self.x, self.fn, self.sigma, key=subkey2)
                
        if not trajectory:
            def run_single_wrapper(data_y_element):
                p_vgd, _, p_svgd, _, _, _ = self.run_single_experiment(data_y_element, initial_particles)
                return p_vgd, p_svgd
        else:
            def run_single_wrapper(data_y_element):
                return self.run_single_experiment(data_y_element, initial_particles)

        if parallel:
            runner = vmap(run_single_wrapper, in_axes=(0,))
            results = runner(dataset)
        else:
            results = lax.map(run_single_wrapper, dataset)

        if not trajectory:
            self.all_particles_VGD, self.all_particles_SVGD = results
            self.all_history_VGD = None
            self.all_history_SVGD = None
            self.all_history_KGD = None
            self.all_history_KSD = None
        else:
            self.all_particles_VGD, self.all_history_VGD, self.all_particles_SVGD, self.all_history_SVGD, self.all_history_KGD, self.all_history_KSD = results


    def plot_diagnostic(self, num_sample_from_posterior = 100, parallel=True, trajectory=True):
        # Run all experiments
        self.resample_experiment(num_sample_from_posterior, parallel=parallel, trajectory=trajectory)
        
        # Compute MMD values for each experiment
        vmapped_mmd = vmap(calculate_mmd_squared, 
                        in_axes=(0,
                                0,             
                                None,  
                                None,  
                                None,         
                                None,  
                                None))
        
        print("MMD length scale:", self.mmd_length_scale)

        mmd_p = 1

        self.all_mmd_values = vmapped_mmd(
            self.all_particles_SVGD,  
            self.all_particles_VGD,
            self.x,
            self.fn,           
            self.mmd_length_scale,    
            self.sigma,           
            mmd_p                
        )

        self.all_mmd_values.block_until_ready()
        
        print("Actual mmd", self.actual_mmd)
        # plt.hist(all_mmd_values)
        # sns.kdeplot(all_mmd_values, fill=True, label='MMD (KDE)', clip=(0, None))
        plt.hist(self.all_mmd_values, bins=30, density=True, alpha=0.6, color='g')
        plt.axvline(
            x=self.actual_mmd, 
            color='red', 
            linestyle='--', 
            linewidth=2, 
            label=f'Actual MMD: {self.actual_mmd:.4f}'
        )
        return self.all_mmd_values, self.actual_mmd        