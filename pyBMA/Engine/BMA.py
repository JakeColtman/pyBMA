from pyBMA.pyBMA.Engine.Selector import Selector

class BMA:

    def __init__(self, model, selector = Selector(), priors = None):
        self.model = model
        self.selector = selector

        if priors is None:
            # If no given prior choose an uniformative one
            self.priors = [0.5] * (len(self.model.get_covariates()) - 2)

    def run(self):
        covariates = self.model.get_covariates()
        models = self.selector.select(covariates)
        results = []
        for model in models:
            print(model)
            print(self.model.run_with_covariates(model))
