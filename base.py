class BaseDER: 
    def __init__(self, pool_regressors=None, k=7, knn_metric='minkowski', metrics='mse'): 
        self.pool_regressors = pool_regressors 
        self.k               = k 
        self.knn_metric      = knn_metric 

        if metrics == 'mse': 
            self.eval_metric = mean_squared_error 


    def get_region_of_competence(self, query): 
        nbrs    = NearestNeighbors(n_neighbors=self.k, metric=self.knn_metric).fit(self.X_dsel)         
        indices = nbrs.kneighbors(query.values.reshape(1, -1), return_distance=False) 

        self.roc = self.X_dsel.iloc[indices[0]] 
        self.roc_labels = self.y_dsel.iloc[indices[0]] 
        

    def estimate_competence(self):
        competence_list = [] 
        
        for regressor in self.pool_regressors: 
            preds = regressor.predict(self.roc) 
            competence = self.eval_metric(self.roc_labels, preds) 
            competence_list.append(competence) 

        return competence_list 


class DER(BaseDER): 
    def __init__(self, pool_regressors=None, k=7, knn_metric='minkowski', metrics='mse'):
        super(DER, self).__init__(pool_regressors=pool_regressors, k=k, knn_metric=knn_metric, metrics=metrics) 
        

    def fit(self, X_dsel=None, y_dsel=None):
        self.X_dsel = X_dsel 
        self.y_dsel = y_dsel  

    
    def select(self):
        pass 


    def predict(self):
        pass 
