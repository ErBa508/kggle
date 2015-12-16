from time import time
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier

class BoostClassifier(object):
    """
    class Boost implements a boost model.
    
    Set: 
    number of trees - unlike bagging & random forests, boosting can overfit if # of trees is too large, 
     although this occurs slowly. Use cross-val to select # of trees.
    learning rate - this controls the rate at which boosting learns. Typical values are 0.01 and 0.001, 
     and the right choice depends on the problem. Very small learning rate requires using larger # of trees. 
    number of splits (d) - often d = 1 works well (a stump). In this case, the boosted ensemble is fitting 
     an additive model, since each term involves only a single variable. 
    """
    
    def __init__(self, X_train, y_train):
        """ """
        self.X_train = X_train
        self.y_train = y_train
        
    def gradientBoost(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, \
                 min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, \
                 max_depth=3, init=None, random_state=None, max_features=None, verbose=0, \
                 max_leaf_nodes=None, warm_start=False, presort='auto'):
        
        start = time()
        # fit estimator
        alg = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, \
                                         subsample=subsample, min_samples_split=min_samples_split, \
                                         min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,\
                                         max_depth=max_depth, init=init, random_state=random_state, max_features=max_features, \
                                         verbose=verbose, max_leaf_nodes=max_leaf_nodes, warm_start=warm_start, presort=presort)
        alg.fit(X_train, y_train)

        print("\nTime elapsed (s) is:", (time() - start)/60)
        
        return alg
        
    def adaBoost(self, base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
        
        start = time()
        # fit estimator
        alg = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, \
                                 algorithm=algorithm, random_state=random_state)
        alg.fit(X_train, y_train)

        print("\nTime elapsed (s) is:", (time() - start)/60)
        
        return alg
        
    def results(self, alg, Xtest, ytest):
        pred = alg.predict(Xtest)
        acc = alg.score(Xtest, ytest)
        print("Best accuracy:", acc)
        
        return pred
