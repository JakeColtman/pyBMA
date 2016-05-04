from lifelines.datasets import load_rossi

from pyBMA.CoxPHFitter import CoxPHFitter

##Replication of http://lifelines.readthedocs.io/en/latest/Survival%20Regression.html

bmaCox = CoxPHFitter()
print(bmaCox.fit(load_rossi(), "week", event_col= "arrest").summary)
