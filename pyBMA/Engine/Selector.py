from itertools import combinations

class Selector:

    def select(self, covariates):
        all_models = []
        for i in range(1, len(covariates)):
            all_models.append(list(combinations(covariates, i)))
        all_models = [list(item) for sublist in all_models for item in sublist]
        return all_models

if __name__ == "__main__":
    print(Selector().select([1, 2, 3]))