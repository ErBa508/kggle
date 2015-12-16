from time import time
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

class searchHyperparameters(object):
    """
    Search a model's hyperparameters using either RandomizedSearchCV or
    GridSearchCV. Print duration of the grid search and a 'results' method
    reveals the best parameters.
    
    Example usage:
        # instantiate algorithm
        alg = GradientBoostingClassifier()
        # define parameter values to be searched
        n_estimators = [1, 3, 5, 10]
        max_depth = [2, ]
        max_features = [1, 4, 9, 12]
        subsample = [0.001, ]
        param_grid = dict(n_estimators = n_estimators, max_depth = max_depth, \
                         max_features = max_features, subsample = subsample)
        # instantiate searchHyperparameters()
        searchHP = searchHyperparameters(alg, param_grid, cval = None, score = None)
        # run grid search
        grid = searchHP.randomGrid(2, X_train, y_train)
        results = searchHP.results(grid)
    """
    def __init__(self, alg, param_grid, cval, score):
        """Give it an instantiated algorithm """
        self.alg = alg
        self.param_grid = param_grid
        self.cval = cval
        self.score = score
        
    def randomGrid(self, n_iter, X, y):
        start = time()
        grid = RandomizedSearchCV(self.alg, self.param_grid, n_iter = n_iter, cv = self.cval, scoring = self.score)
        grid.fit(X,y)
        print("\nTime elapsed (s) is:", (time() - start)/60)
        return grid
    
    def fullGrid(self, X, y):
        start = time()
        grid = GridSearchCV(self.alg, self.param_grid, cv = self.cval, scoring = self.score)
        grid.fit(X,y)
        print("\nTime elapsed (s) is:", (time() - start)/60)
        return grid
    
    def results(self, grid):
        print("Overall results:", grid.grid_scores_)
        print("Best score:", grid.best_score_)
        print("Best parameters:", grid.best_params_)
        print("Best model:", grid.best_estimator_)

        # note if SD high, cross-val estimates may not be reliable
        results = grid.grid_scores_
        return results