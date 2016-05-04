from lifelines.datasets import load_rossi
from pyBMA.CoxPHFitter import CoxPHFitter

# Replication of http://lifelines.readthedocs.io/en/latest/Survival%20Regression.html

bmaCox = CoxPHFitter()
print(bmaCox.fit(load_rossi(), "week", event_col= "arrest").summary)

# If you wanted to have a model with few variables
bmaCox = CoxPHFitter()
print(bmaCox.fit(load_rossi(), "week", event_col= "arrest", priors= [0.2]*7).summary)

# If you wanted to have a model with many variables
bmaCox = CoxPHFitter()
print(bmaCox.fit(load_rossi(), "week", event_col= "arrest", priors= [0.8]*7).summary)

# If you're very confident that race has no impact
# rossi columns:  fin 	age 	race 	wexp 	mar 	paro 	prio
bmaCox = CoxPHFitter()
print(bmaCox.fit(load_rossi(), "week", event_col= "arrest", priors= [0.5, 0.5, 0.1, 0.5, 0.5, 0.5, 0.5]).summary)

# If you're very confident that both age and race should be included
# rossi columns:  fin 	age 	race 	wexp 	mar 	paro 	prio
bmaCox = CoxPHFitter()
print(bmaCox.fit(load_rossi(), "week", event_col= "arrest", priors= [0.5, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5]).summary)
