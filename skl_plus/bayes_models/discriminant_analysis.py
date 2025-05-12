import numpy as np

def _covariance(X, y, pooled=True):

    classes = np.unique(y)
    print(X.shape)
    n_features = X.shape[1]

    if pooled: 
        cov = np.zeros((n_features, n_features)) 
    else: 
        cov = np.zeros((len(classes), n_features, n_features)) 

    mean_vector = np.zeros((X.shape[1], len(classes)))

    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        Xg_mean = np.mean(Xg, axis=0)
        Xg_centered = Xg - Xg_mean  
        
        mean_vector[:, idx] = Xg_mean
        
        cov_k = (Xg_centered.T @ Xg_centered) / (Xg.shape[0] - 1)
        
        if pooled:
            cov += cov_k
        else: 
            cov[idx] = cov_k

    if pooled: 
        cov /= (X.shape[0] - len(classes))
        
    return cov , mean_vector
    

def _softmax(score):

    np.set_printoptions(suppress=True, precision=8)
    exp_score = np.exp(score - np.max(score, axis=1, keepdims=True))
    return exp_score / np.sum(exp_score, axis=1, keepdims=True)


class LDAClassifier:
    """
    Linear Discriminant Analysis (LDA) Classifier

    Parameters:
    -----------
    priors: list or none
        List of class priors. If None, computed from data.
    
    use_class_weight: bool, default=False   
        Whether to use pass class weights. 
    
    class_weight: dict or None 
        Dictionary specifying class weights. Required, if use_class_weight is True
    
    unbalanced_adjustment: bool, default=True
        Experimental. Adjusts Discriminant function to boost underrepresented class.
    """
    def __init__(
            self,
            priors:list = None, 
            use_class_weight:bool = False,
            class_weight:dict = None,
            unbalanced_adjustment = False
    ):
        self.priors = priors
        self.use_class_weight = use_class_weight
        self.class_weight = class_weight
        self.unbalanced_adjustment = unbalanced_adjustment


        if self.use_class_weight and self.class_weight is None:
            message = "Pass class weights in class_weight or set use_class_weight=False"
            raise ValueError(message)
        
    def _discriminant(self, X):
        proba = np.zeros((X.shape[0], len(self.classes_) ))

        inv_cov = np.linalg.pinv(self.cov)

        for idx, cls in enumerate(self.classes_):
            mean_vec = self.class_mean_[:, idx]
            term1 = X @ inv_cov @ mean_vec
            term2 = 0.5 * mean_vec.T @ inv_cov @ mean_vec
            term3 = np.log(self.priors[idx]) 

            proba[:, idx] = term1 - term2 + term3

        return proba


    def fit(self, X, y):
        print('Reloaded')

        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        if self.priors is None:
            self.priors = self.counts_ / len(y)

        self.cov, self.class_mean_ = _covariance(X, y, pooled=True)

        return self
    
    def predict(self, X):

        discriminant = self._discriminant(X)
        class_indices = np.argmax(discriminant, axis=1)
        preds = (self.classes_[class_indices])
 
        return preds
    
    def predict_proba(self, X):

        discriminant = self._discriminant(X)

        return _softmax(discriminant)



class QDAClassifier:
    """
    Discriminant Discriminant Analysis (QDA) Classifier

    Parameters:
    -----------
    priors: list or none
        List of class priors. If None, computed from data.
    
    use_class_weight: bool, default=False   
        Whether to use pass class weights. 
    
    class_weight: dict or None 
        Dictionary specifying class weights. Required, if use_class_weight is True
    
    unbalanced_adjustment: bool, default=True
        Experimental. Adjusts Discriminant function to boost underrepresented class.
    """
    def __init__(
            self,
            priors:list = None, 
            use_class_weight:bool = False,
            class_weight:dict = None,
            unbalanced_adjustment = False
    ):
        self.priors = priors
        self.use_class_weight = use_class_weight
        self.class_weight = class_weight
        self.unbalanced_adjustment = unbalanced_adjustment


        if self.use_class_weight and self.class_weight is None:
            message = "Pass class weights in class_weight or set use_class_weight=False"
            raise ValueError(message)
        
    def _discriminant(self, X):
        proba = np.zeros((X.shape[0], len(self.classes_) ))


        for idx, cls in enumerate(self.classes_):

            mean_vec = self.class_mean_[:, idx]
            cov_k = self.cov[idx]
            epsilon = 1e-6
            cov_k += np.eye(cov_k.shape[0]) * epsilon
            inv_cov = np.linalg.pinv(cov_k)

            X_centered = X - mean_vec
            term1 = -0.5 * np.sum(X_centered @ inv_cov * X_centered, axis=1)
            
            sign, logdet = np.linalg.slogdet(cov_k)
            if sign <= 0:
                raise ValueError(f"Covariance matrix for class {cls} is not positive definite.")
            term2 = -0.5 * logdet            
            
            term3 = np.log(self.priors[idx])

            proba[:, idx] = term1 + term2 + term3 

        return proba


    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        if self.priors is None:
            self.priors = self.counts_ / len(y)

        self.cov, self.class_mean_ = _covariance(X, y, pooled=False)

        return self

    
    def predict(self, X):

        discriminant = self._discriminant(X)
        class_indices = np.argmax(discriminant, axis=1)
        preds = (self.classes_[class_indices]).T
 
        return preds

    def predict_proba(self, X):
        
        discriminant = self._discriminant(X)
        
        return _softmax(discriminant)
