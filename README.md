# pyBMA

Bayesian Model Averaging in python

This module is based on the R package BMA and implements Bayesian Model Averaging for the cox proportional hazards model.  

#### Installation

pyBMA can be installed from pypi using pip as normal

	pip3.5 install pyBMA
	
#### How it works

Given a survial dataset, pyBMA does the following things:

1 - Uses a leaps and bounds algorithm to sample model space

2 - Uses [lifelines](http://lifelines.readthedocs.io/en/latest/Survival%20Regression.html#cox-s-proportional-hazard-model) to run Cox Proprtional hazards and generate log-likihood.

3 - Calculates the posterior likihood of the model given the data and some priors

4 - Performs a weighted average over the models based on the posterior model likihood
	
#### How to use it

The API of pyBMA is designed to mirror that of lifelines to allow as easy an integration as possible.  Compare the two in the snippet below:

``` python

##pyBMA version
bma_cf = CoxPHFitter()
bma_cf.fit(rossi_dataset, 'week', event_col='arrest')

## Lifelines version
cf = CoxPHFitter()
cf.fit(rossi_dataset, 'week', event_col='arrest')

```

One addition is that you can now specify a prior for each variable.  This should be inputted as a numpy array of numbers between 0 and 1 in the same order as the covariate variables appear in the main dataframe.  The prior for a variable is your belief about the probability that the variable will be included in the correct model.  E.g. if you are certain that a variable must occur in a model for it to be correct, then set the prior for that variable to 1, while if you consider it as likely as not to be included then choose 0.5.  The default sets all the priors to 0.5


``` python

##pyBMA version
bma_cf = CoxPHFitter()
bma_cf.fit(rossi_dataset, 'week', event_col='arrest', priors = np.array([0.3, 0.6, 0.7, 0.1, 0.9, 0.5, 0.03])

```

This setup will give more weight to those models which contain the 5th variable, and much less weight to the models including the last variable over and above the model's likihood given just the data.

More examples can be found in lifelines_example.py


