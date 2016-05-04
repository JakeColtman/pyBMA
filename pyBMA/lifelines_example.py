from lifelines.datasets import load_rossi

from pyBMA.CoxPHFitter import CoxPHFitter

##Replication of http://lifelines.readthedocs.io/en/latest/Survival%20Regression.html

bmaCox = CoxPHFitter()
posterior = bmaCox.fit(load_rossi(), "week", event_col= "arrest")
print(posterior[0])