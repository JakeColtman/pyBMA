from Survival.CoxPH import CoxPH
from lifelines.datasets import load_rossi

##Replication of http://lifelines.readthedocs.io/en/latest/Survival%20Regression.html

bmaCox = CoxPH(load_rossi(), "week", "arrest")
posterior = bmaCox.run()
print(posterior[0])