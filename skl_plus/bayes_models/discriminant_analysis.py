import numpy as np
from sklearn.exceptions import NotFittedError
from collections import defaultdict
import pprint

def _covariance(X, y, pooled=True):

    classes = np.unique(y)
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

        self.fitted = False

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

        self.fitted = True

        return self
    
    def predict(self, X):

        if not self.fitted:
            msg = "Call fit() atleast once, before calling predict()"
            raise NotFittedError(msg)

        discriminant = self._discriminant(X)
        class_indices = np.argmax(discriminant, axis=1)
        preds = (self.classes_[class_indices])
 
        return preds
    
    def predict_proba(self, X):

        if not self.fitted:
            msg = "Call fit() atleast once, before calling predict()"
            raise NotFittedError(msg)
        
        discriminant = self._discriminant(X)

        return _softmax(discriminant)



class QDAClassifier:
    """
    Quadratic Discriminant Analysis (QDA) Classifier

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
        self.fitted = False


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

        self.fitted = True

        return self

    
    def predict(self, X):

        if not self.fitted:
            msg = "Call fit() atleast once, before calling predict()"
            raise NotFittedError(msg)
        
        discriminant = self._discriminant(X)
        class_indices = np.argmax(discriminant, axis=1)
        preds = (self.classes_[class_indices]).T
 
        return preds

    def predict_proba(self, X):
        
        if not self.fitted:
            msg = "Call fit() atleast once, before calling predict()"
            raise NotFittedError(msg)

        discriminant = self._discriminant(X)
        
        return _softmax(discriminant)

class CategoricalDA:
    """
    Experimental Discriminant Analysis Classifier for Categorical data
    --- Don't use this model! This has bugs, that I don't wanna debug. Peace ---
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
            pooled:str = False,
            use_class_weight:bool = False,
            class_weight:dict = None,
            unbalanced_adjustment = False
    ):
        self.priors = priors
        self.use_class_weight = use_class_weight
        self.class_weight = class_weight
        self.unbalanced_adjustment = unbalanced_adjustment
        self.fitted = False
        self.pooled = pooled


        if self.use_class_weight and self.class_weight is None:
            message = "Pass class weights in class_weight or set use_class_weight=False"
            raise ValueError(message)

    def _numeric_cov(self):

        X = np.asarray(self.X.iloc[:, self.num_cols], dtype=np.float64)
        n_features = len(self.features)
        num_features = len(self.num_cols)

        if self.pooled: 
            cov = np.zeros((num_features, num_features)) 
            full_cov = np.zeros((n_features, n_features)) 
        else: 
            cov = np.zeros((len(self.classes_), num_features, num_features))
            full_cov = np.zeros((len(self.classes_), n_features, n_features))
        numeric_indices = self.num_cols
        mean_vector = np.zeros((n_features, len(self.classes_)))
        mean = np.zeros((X.shape[1], len(self.classes_)))
        # Compute covariance for numeric columns only
        for idx, group in enumerate(self.classes_):
            Xg = X[self.y == group, :]
            Xg_mean = np.mean(Xg, axis=0)
            Xg_centered = Xg - Xg_mean  
            
            mean[:, idx] = Xg_mean
            print(Xg_mean)
            cov_k = (Xg_centered.T @ Xg_centered) / (Xg.shape[0] - 1)
            
            if self.pooled:
                cov += cov_k
            else: 
                cov[idx] = cov_k        
            # Filling the mean Vector and covariance matrix with zeros
            # Place numeric covariances in the right spots
            for i, row in enumerate(numeric_indices):
                mean_vector[row, idx] = mean[i, idx]
                for j, col in enumerate(numeric_indices):
                    if self.pooled:
                        full_cov[row, col] = cov[i, j]
                    else:
                        full_cov[idx, row, col] = cov[idx, i, j]
        print(mean_vector)
        return full_cov, mean_vector
        
    def _cat_cov(self):

        cat_joint_probs = defaultdict(lambda: defaultdict(dict))  # class → (f1,f2) → (v1,v2) → prob
        cat_mean_k = defaultdict(lambda: defaultdict(dict))
        cat_mean = defaultdict(dict)
        if self.pooled:
            combined_prob = defaultdict(dict)
        for k in self.classes_:
            X_k = self.X[self.y == k]
            for i in range(len(self.features)):
                for j in range(len(self.features)):

                    # Computing pairwise Join Probability for columns, works as Psuedo-Covariance 
                    if i != j:
                        col1, col2 = X_k.columns[i], X_k.columns[j]
                        
                        # If both columns are Categorical, Cat-Cat joint probability
                        if X_k[[col1, col2]].dtypes.tolist() == ['category', 'category']:   
                            cross_counts = X_k.groupby([col1, col2], observed=False).size().div(len(X_k))
                            for (v1, v2), p in cross_counts.items():
                                cat_joint_probs[k][(i, j)][(v1, v2)] = p

                                # Combining Probabilities if the estimator is set to 'lda'
                                if self.pooled:
                                    if (v1, v2) in combined_prob[(col1, col2)]:
                                        combined_prob[(i, j)][(v1, v2)] += len(X_k) * cat_joint_probs[k][(i, j)][(v1, v2)]
                                    else:
                                        combined_prob[(i, j)][(v1, v2)] = len(X_k) * cat_joint_probs[k][(i, j)][(v1, v2)]
                        elif ('category' in X_k[[col1, col2]].dtypes.tolist()):
                            cat_col = col1 if X_k[col1].dtype == 'category' else col2
                            num_col = col1 if X_k[col1].dtype != 'category' else col2

                            mean = np.mean(X_k[num_col], axis=0)
                            var = (sum((X_k[num_col] - mean)**2)) / (X_k.shape[0] - 1)

                            for cat in np.unique(X_k[cat_col]):
                                prob =  len(X_k[X_k[cat_col]==cat]) / X_k.shape[0]
                                cat_joint_probs[k][(i, j)][(cat, 'var')] = var * prob
                                
                                if self.pooled:
                                    if (cat, 'var') in  combined_prob[(i, j)]:
                                        combined_prob[(i, j)][(cat, 'var')] += len(X_k) * cat_joint_probs[k][(i, j)][(cat, 'var')]
                                    else:
                                        combined_prob[(i, j)][(cat, 'var')] = len(X_k) * cat_joint_probs[k][(i, j)][(cat, 'var')]

                    #  Probability of a Category to be in class_k, works as Psuedo-Variance
                    else:
                        col = X_k.columns[i]
                        if 'category' == X_k[col].dtypes:
                            print(col)
                            for cat in np.unique(X_k[col]): 
                                p = len(X_k[X_k[col]==cat]) / X_k.shape[0]
                                cat_joint_probs[k][(i, i)][(cat, cat)] = p
                                cat_mean_k[k][i][cat] = p 
                                cat_mean[i][cat] = p + cat_mean[i].get(cat, 0)
                                if self.pooled:
                                    col1, col2 = self.features[i], self.features[j]
                                    if (cat, cat) in combined_prob[(i, i)]:
                                        combined_prob[(i, i)][(cat, cat)] += len(X_k) * cat_joint_probs[k][(i, j)][(cat, cat)]
                                    else:
                                        combined_prob[(i, i)][(cat, cat)] = len(X_k) * cat_joint_probs[k][(i, j)][(cat, cat)]

        # Normalizing each probability if estimator is 'lda'
        if self.pooled:
            for col in  combined_prob.items():
                for prob in col[1].items():
                        combined_prob[col[0]][prob[0]] = prob[1] / len(self.X) 
        cat_cov =  combined_prob if self.pooled else cat_joint_probs
        cat_cov = {k: v for k, v in cat_cov.items() if v}
        pprint.pprint(cat_mean)
        pprint.pprint(cat_mean_k)
        return cat_cov, cat_mean_k, cat_mean
                                
    def _fill_cov(self, X, cat_cov, cat_mean, full_cov, mean_vector):
        
        imp_x = X.copy()
        for (key, value) in list(cat_cov.items()):
            xi = []
            for i in key:
                xi.append(X[i])
            _dtype = [type(x) for x in xi]
            if _dtype == [str, str]:
                p = value[tuple(xi)]
                full_cov[key[0], key[1]] = p
                if key[0] == key[1]:
                    mean_vector[key[0]] = cat_mean[key[0]][xi[0]]
                    imp_x[key[0]] = self.cat_mean[key[0]][X[key[0]]]

            elif str in _dtype:
                cat = xi[0] if type(xi[0]) == str else xi[1]
                p = value[(cat, 'var')]
                full_cov[key[0], key[1]] = p
                        # print('ys')
        

        return imp_x, full_cov, mean_vector
    
    def _qda(self, X):

        proba = np.zeros((X.shape[0], len(self.classes_) ))
        for i, xi in enumerate(X.itertuples()):
            for idx, cls in enumerate(self.classes_):
                print(xi)
                mean_vec = self.class_mean_[:, idx]
                imp_xi, cov_k, mean_vec = self._fill_cov(list(xi)[1:], self.cat_cov[cls], self.cat_mean_k[cls], self.full_cov[idx], mean_vec)
                epsilon = 0.1
                print(cov_k)
                cov_k += np.eye(cov_k.shape[0]) * epsilon
                inv_cov = np.linalg.pinv(cov_k)
                X_centered = imp_xi - mean_vec
                
                term1 = -0.5 * np.sum(X_centered @ inv_cov * X_centered, axis=0)

                sign, logdet = np.linalg.slogdet(cov_k)
                if sign <= 0:
                    print("ERRRRROOOORRRRRR")
                    print(cov_k)
                    print(logdet) 
                    raise ValueError(f"Covariance matrix for class {cls} is not positive definite.")
                term2 = -0.5 * logdet            
                
                term3 = np.log(self.priors[idx])

                proba[i, idx] = term1 + term2 + term3 
        
        print(proba)
        return proba           
    def fit(self, X, y):

        self.X = X.copy()
        self.y = np.asarray(y.copy())

        self.features = self.X.columns.tolist()
        self.cat_cols = [self.X.columns.get_loc(c) for c in self.X.select_dtypes(include='category').columns]
        self.num_cols = [self.X.columns.get_loc(c) for c in self.X.select_dtypes(exclude='category').columns]

        self.classes_, self.counts_ = np.unique(y, return_counts=True)
        
        if self.priors == None:
            self.priors = self.counts_ / len(self.y)
        print(self.priors)
        self.full_cov, self.class_mean_ = self._numeric_cov()
        self.cat_cov, self.cat_mean_k, self.cat_mean = self._cat_cov()
        self.fitted = True
        # pprint.pprint(self.cat_cov)
        return self
    
    def predict(self, X):

        if not self.fitted:
            msg = "Call fit() atleast once, before calling predict()"
            raise NotFittedError(msg)

        discriminant = self._qda(X)
        class_indices = np.argmax(discriminant, axis=1)
        preds = (self.classes_[class_indices])
        print(preds)
        return 