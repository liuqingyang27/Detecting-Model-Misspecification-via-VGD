# Detecting Model Misspecification in Bayesian Inverse Problems via Variational Gradient Descent
Link to this article on arxiv: https://arxiv.org/abs/2512.01667

Experiments are carried out by the notebooks in the folder experiments. Figure 1, 2 and 5 can be reproduced in main fig.ipynb. Figure 6 can be reproduced in different data size.ipynb. Figure 7 can be reproduced in different dimension fig sine.ipynb.
Algorithms and all other functions are contained in the folder func.
- calculate_mmd: function to calculate squared mmd for diagnostic
- experiment: class to run experiments and diagnostics
- kernel: kernel functions
- method: VGD algorithm
- model: class to realise a model, including prior, likelihood, noise level and the dimension of the parameter
- plot_functions: functions for plotting
