from math import exp

from . import CoxPHModel


class CoxPH:
    def __init__(self, x, survival_col, cens_col, priors=None):
        self.df = x
        self.survival_col = survival_col
        self.cens_col = cens_col
        if priors == None:
            self.priors = [0.5] * (len(self.df.columns) - 2)  # Uniformative prior
        else:
            self.priors = priors
        self.reference_loglik = None
        self.full_model = self.create_model(None)
        self._set_reference_loglik()

    def _generate_model_definnitions(self):
        names, coefs, var = self.full_model.summary()
        model1 = ["fin", "prio"]
        model2 = ["race", "mar"]
        model6 = ["race", "age"]
        model3 = ["prio", "race"]
        model4 = ["prio", "race", "mar"]
        model5 = ["prio", "age", "mar"]
        return [model1, model2, model3, model4, model5, model6]

    def _weight_by_posterior(self, values, posterior):
        def add_dataframes(dfone, dftwo):
            return dfone.add(dftwo, fill_value=0)

        output = zip(values, posterior)
        weighted = [x[0] * x[1] for x in output]
        running_total = weighted[0]
        for i in range(1, len(weighted)):
            running_total = add_dataframes(running_total, weighted[i])
        return running_total

    def run(self):

        models = self._generate_model_definnitions()
        models = [self.create_model(x) for x in models]
        bics = [x.bayesian_information_critera() for x in models]
        self.posterior_probabilities = []
        min_bic = min(bics)
        summation = sum([exp(-0.5 * (bic - min_bic)) for bic in bics])
        for bic in bics:
            posterior = (exp(-0.5 * (bic - min_bic))) / summation
            self.posterior_probabilities.append(posterior)

        coefficiencts_by_model = [x.summary()[1] for x in models]
        sterr_by_model = [x.summary()[2] for x in models]

        self.coefficients_weighted = self._weight_by_posterior(coefficiencts_by_model, self.posterior_probabilities)
        self.sterr_weighted = self._weight_by_posterior(sterr_by_model, self.posterior_probabilities)
        return self.coefficients_weighted, self.sterr_weighted

    def create_model(self, covariate_names):
        return CoxPHModel.CoxPHModel(self.df, self.survival_col, self.cens_col, self.priors, self.reference_loglik,
                                     covariate_names)

    def _set_reference_loglik(self):
        self.reference_loglik = self.full_model.loglik()
