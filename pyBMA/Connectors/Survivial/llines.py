from pyBMA.pyBMA.Engine.BMA import BMA
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter

class LlinesModel:
    def __init__(self, full_data_set, event_col, duration_col):
        self.data_set = full_data_set
        self.event_col = event_col
        self.duration_col = duration_col

    def run_with_covariates(self, covariates = None):
        if covariates is None:
            ds = self.data_set
        else:
            ds = self.data_set[[self.event_col, self.duration_col] + covariates]

        cf = CoxPHFitter()
        cf.fit(ds, self.duration_col, event_col=self.event_col, include_likelihood = True)
        print(cf._log_likelihood)

    def get_covariates(self):
        return list(filter(lambda x: x not in [self.event_col, self.duration_col], self.data_set.columns.values))

if __name__ == "__main__":
    rossi_dataset = load_rossi()

    model = LlinesModel(rossi_dataset, 'arrest', 'week')
    bma = BMA(model)
    bma.run()