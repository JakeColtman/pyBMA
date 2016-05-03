import lifelines
from functools import reduce
from math import log
import operator

class CoxPHModel:
    def __init__(self, df, survival_col, cens_col, prior_params, reference_loglik=None, covariate_names=None):
        self.prior_params = prior_params
        self.survival_col = survival_col
        self.cens_col = cens_col

        all_covariate_columns = [col for col in df.columns if col not in [cens_col, survival_col]]
        if covariate_names == None:
            self.covariate_names = all_covariate_columns
        else:
            self.covariate_names = covariate_names
        self.df = df[self.covariate_names + [self.survival_col, self.cens_col]]

        self.mask = [x in self.covariate_names for x in all_covariate_columns]
        self._cf = None
        if reference_loglik == None:
            reference_loglik = self.loglik()
        self.reference_loglik = reference_loglik

    def prior(self):
        parameter_contributions = [x[1] if x[0] else (1 - x[1]) for x in zip(self.mask, self.prior_params)]
        return reduce(operator.mul, parameter_contributions, 1)

    def _run(self):
        self._cf = lifelines.CoxPHFitter()
        self._cf.fit(self.df, self.survival_col, event_col=self.cens_col, include_likelihood=True)

    def loglik(self):
        if self._cf is None:
            self._run()
        return self._cf._log_likelihood

    def summary(self):
        if self._cf is None:
            self._run()
        return self._cf.summary.index, self._cf.summary["coef"], (
            self._cf.summary["se(coef)"] * self._cf.summary["se(coef)"])

    def bayesian_information_critera(self):
        size = len(self.covariate_names)
        n = self.df.shape[0]
        prior = self.prior()
        loglik = self.loglik()
        return (size * log(n)) - (2 * (loglik - self.reference_loglik)) - (2 * log(prior))
