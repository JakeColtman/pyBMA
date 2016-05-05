from math import exp
import pandas as pd
from pyBMA import CoxPHModel
from numpy.linalg import solve, norm, inv
from itertools import combinations

class CoxPHFitter:
    """
    Fitter for Cox Proportional Hazards using Bayesian model averaging:
    h(t|x) = h_0(t)*exp(x'*beta)
    Fitting of individual models is done using lifelines
    """

    def fit(self, df, duration_col, event_col, priors=None):
        """
        Average across models to produce the BMA estimate for coefficients
        Parameters:
          df: a Pandas dataframe.

              Required columns: duration_col and event_col

              Other columns: covariates to model

              duration_col: lifetime of subject in an arbitrary unit.
              event_col: indicator for whether a death event was observerd.
                            1: Death observed
                            0: Censored

          duration_col: name of column holding duration info
          event_col: name of column holding event info

          priors: A list of length = number of covariates.  Indexed by the ordering of covariates in df
                  Each element of the list is the probability of the respective variable being included in a correct model
                    e.g. if you are certain a variable should be included, set this to 1
                    if you wish to encourage parsimonious models set the value for all variables to be < 0.5
                    if you want to encourage complex models, set all values to > 0.5

                  Values should be restricted to [0 -> 1]

                  default: [0.5] * number covariates:
                           completely uninformative, all models considered as likely
        Returns:
            self
        """

        self.df = df
        self.duration_col = duration_col
        self.event_col = event_col

        if priors is None:
            # If no given prior choose an uniformative one
            self.priors = [0.5] * (len(self.df.columns) - 2)
        else:
            self.priors = priors

        self.reference_loglik = None

        # Create a baseline model using all covariates.
        self.full_model = self._create_model(None)
        self._set_reference_loglik()

        # Generate representative sample of model space
        models = self._generate_model_definnitions()
        models = [self._create_model(x) for x in models]

        # Process log likihoods into posterior probabilities
        bics = [x.bayesian_information_critera() for x in models]
        self._generate_posteriors_from_bic(bics)

        coefficiencts_by_model = [x.summary()[1] for x in models]
        self.coefficients_weighted = self._weight_by_posterior(coefficiencts_by_model)
        sterr_by_model = [x.summary()[2] for x in models]
        self.sterr_weighted = self._weight_by_posterior(sterr_by_model)

        return self

    @property
    def summary(self):
        """Details of the output.
        Returns
        -------
        df : pd.DataFrame
            Contains columns coef, exp(coef)"""

        df = self.coefficients_weighted.to_frame()
        df['exp(coef)'] = [exp(x) for x in df['coef']]
        return df

    def _create_model(self, covariate_names):
        return CoxPHModel.CoxPHModel(self.df, self.duration_col, self.event_col, self.priors, self.reference_loglik,
                                     covariate_names)

    def _set_reference_loglik(self):
        self.reference_loglik = self.full_model.loglik()

    def _generate_model_definnitions(self):
        names, coefs, var = self.full_model.summary()
        variance_covariance = inv(-self.full_model._cf._hessian_)
        all_models = []
        for i in range(1, len(names)):
            all_models.append(list(combinations(names, i)))
        all_models = [list(item) for sublist in all_models for item in sublist]
        return all_models

    def _generate_posteriors_from_bic(self, bics):
        self.posterior_probabilities = []
        min_bic = min(bics)
        summation = sum([exp(-0.5 * (bic - min_bic)) for bic in bics])
        for bic in bics:
            posterior = (exp(-0.5 * (bic - min_bic))) / summation
            self.posterior_probabilities.append(posterior)

    def _weight_by_posterior(self, values):
        def add_dataframes(dfone, dftwo):
            return dfone.add(dftwo, fill_value=0)

        output = zip(values, self.posterior_probabilities)
        weighted = [x[0] * x[1] for x in output]
        running_total = weighted[0]
        for i in range(1, len(weighted)):
            running_total = add_dataframes(running_total, weighted[i])
        return running_total
