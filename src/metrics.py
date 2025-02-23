#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
from scipy import linalg, stats
from sklearn.mixture import GaussianMixture 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import iterative_A
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NeighborhoodComponentsAnalysis
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from pytorch_metric_learning import losses
import gc

def _cov(X, shrinkage=-1):
    emp_cov = np.cov(np.asarray(X).T, bias=1)
    if shrinkage < 0:
        return emp_cov
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def softmax(X, copy=True):
    if copy:
        X = np.copy(X)
    max_prob = np.max(X, axis=1).reshape((-1, 1))
    X -= max_prob
    np.exp(X, X)
    sum_prob = np.sum(X, axis=1).reshape((-1, 1))
    X /= sum_prob
    return X


def _class_means(X, y):
    """Compute class means.
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data.
    y : array-like of shape (n_samples,) or (n_samples, n_targets)
        Target values.
    Returns
    -------
    means : array-like of shape (n_classes, n_features)
        Class means.
    means ： array-like of shape (n_classes, n_features)
        Outer classes means.
    """
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]

    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)    
    return means, means_


def split_data(data: np.ndarray, percent_train: float):
    split = data.shape[0] - int(percent_train * data.shape[0])
    return data[:split], data[split:]


def feature_reduce(features: np.ndarray, f: int=None):
    """
        Use PCA to reduce the dimensionality of the features.
        If f is none, return the original features.
        If f < features.shape[0], default f to be the shape.
	"""
    if f is None:
        return features
    if f > features.shape[0]:
        f = features.shape[0]
    
    return PCA(
        n_components=f,
        svd_solver='randomized',
        random_state=1919,
        iterated_power=1).fit_transform(features)


class TransferabilityMethod:	
    def __call__(self, 
        features: np.ndarray, y: np.ndarray,
                ) -> float:
        self.features = features		
        self.y = y
        return self.forward()

    def forward(self) -> float:
        raise NotImplementedError


class PARC(TransferabilityMethod):
	
    def __init__(self, n_dims: int=None, fmt: str=''):
        self.n_dims = n_dims
        self.fmt = fmt

    def forward(self):
        self.features = feature_reduce(self.features, self.n_dims)
        
        num_classes = len(np.unique(self.y, return_inverse=True)[0])
        labels = np.eye(num_classes)[self.y] if self.y.ndim == 1 else self.y

        return self.get_parc_correlation(self.features, labels)

    def get_parc_correlation(self, feats1, labels2):
        scaler = StandardScaler()

        feats1  = scaler.fit_transform(feats1)

        rdm1 = 1 - np.corrcoef(feats1)
        rdm2 = 1 - np.corrcoef(labels2)
        
        lt_rdm1 = self.get_lowertri(rdm1)
        lt_rdm2 = self.get_lowertri(rdm2)
        
        return stats.spearmanr(lt_rdm1, lt_rdm2)[0] * 100

    def get_lowertri(self, rdm):
        num_conditions = rdm.shape[0]
        return rdm[np.triu_indices(num_conditions, 1)]


class SFDA():
    def __init__(self, shrinkage=None, priors=None, n_components=None):
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        
    def _solve_eigen(self, X, y, shrinkage):
        classes, y = np.unique(y, return_inverse=True)
        cnt = np.bincount(y)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, y, X)
        means /= cnt[:, None]
        self.means_ = means
                
        cov = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Xg = X[y == group, :]
            cov += self.priors_[idx] * np.atleast_2d(_cov(Xg))
        self.covariance_ = cov

        Sw = self.covariance_  # within scatter
        if self.shrinkage is None:
            # adaptive regularization strength
            largest_evals_w = iterative_A(Sw, max_iterations=3)
            shrinkage = max(np.exp(-5 * largest_evals_w), 1e-10)
            self.shrinkage = shrinkage
        else:
            # given regularization strength
            shrinkage = self.shrinkage
        print("Shrinkage: {}".format(shrinkage))
        # between scatter
        St = _cov(X, shrinkage=self.shrinkage) 

        # add regularization on within scatter   
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu

        Sb = St - shrunk_Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def fit(self, X, y):
        '''
        X: input features, N x D
        y: labels, N

        '''
        self.classes_ = np.unique(y)
        #n_samples, _ = X.shape
        n_classes = len(self.classes_)

        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
        self.priors_ = np.bincount(y_t) / float(len(y))
        self._solve_eigen(X, y, shrinkage=self.shrinkage,)

        return self
    
    def transform(self, X):
        # project X onto Fisher Space
        X_new = np.dot(X, self.scalings_)
        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        scores = np.dot(X, self.coef_.T) + self.intercept_
        return softmax(scores)


def each_evidence(y_, f, fh, v, s, vh, N, D):
    """
    compute the maximum evidence for each class
    """
    epsilon = 1e-5
    alpha = 1.0
    beta = 1.0
    lam = alpha / beta
    tmp = (vh @ (f @ np.ascontiguousarray(y_)))
    for _ in range(11):
        # should converge after at most 10 steps
        # typically converge after two or three steps
        gamma = (s / (s + lam)).sum()
        # A = v @ np.diag(alpha + beta * s) @ v.transpose() # no need to compute A
        # A_inv = v @ np.diag(1.0 / (alpha + beta * s)) @ v.transpose() # no need to compute A_inv
        m = v @ (tmp * beta / (alpha + beta * s))
        alpha_de = (m * m).sum()
        alpha = gamma / (alpha_de + epsilon)
        beta_de = ((y_ - fh @ m) ** 2).sum()
        beta = (N - gamma) / (beta_de + epsilon)
        new_lam = alpha / beta
        if np.abs(new_lam - lam) / lam < 0.01:
            break
        lam = new_lam
    evidence = D / 2.0 * np.log(alpha) \
               + N / 2.0 * np.log(beta) \
               - 0.5 * np.sum(np.log(alpha + beta * s)) \
               - beta / 2.0 * (beta_de + epsilon) \
               - alpha / 2.0 * (alpha_de + epsilon) \
               - N / 2.0 * np.log(2 * np.pi)
    return evidence / N, alpha, beta, m


def truncated_svd(x):
    u, s, vh = np.linalg.svd(x.transpose() @ x)
    s = np.sqrt(s)
    u_times_sigma = x @ vh.transpose()
    k = np.sum((s > 1e-10) * 1)  # rank of f
    s = s.reshape(-1, 1)
    s = s[:k]
    vh = vh[:k]
    u = u_times_sigma[:, :k] / s.reshape(1, -1)
    return u, s, vh


class LogME(object):
    def __init__(self, regression=False):
        """
            :param regression: whether regression
        """
        self.regression = regression
        self.fitted = False
        self.reset()

    def reset(self):
        self.num_dim = 0
        self.alphas = []  # alpha for each class / dimension
        self.betas = []  # beta for each class / dimension
        # self.ms.shape --> [C, D]
        self.ms = []  # m for each class / dimension

    def _fit_icml(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the ICML 2021 paper
        "LogME: Practical Assessment of Pre-trained Models for Transfer Learning"
        at http://proceedings.mlr.press/v139/you21b.html
        """
        fh = f
        f = f.transpose()
        D, N = f.shape
        v, s, vh = np.linalg.svd(f @ fh, full_matrices=True)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            evidence, alpha, beta, m = each_evidence(y_, f, fh, v, s, vh, N, D)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    def _fit_fixed_point(self, f: np.ndarray, y: np.ndarray):
        """
        LogME calculation proposed in the arxiv 2021 paper
        "Ranking and Tuning Pre-trained Models: A New Paradigm of Exploiting Model Hubs"
        at https://arxiv.org/abs/2110.10545
        """
        # k = min(N, D)
        N, D = f.shape  

        # direct SVD may be expensive
        if N > D: 
            u, s, vh = truncated_svd(f)
        else:
            u, s, vh = np.linalg.svd(f, full_matrices=False)
        # u.shape = N x k, s.shape = k, vh.shape = k x D
        s = s.reshape(-1, 1)
        sigma = (s ** 2)

        evidences = []
        self.num_dim = y.shape[1] if self.regression else int(y.max() + 1)
        for i in range(self.num_dim):
            y_ = y[:, i] if self.regression else (y == i).astype(np.float64)
            y_ = y_.reshape(-1, 1)
            
            # x has shape [k, 1], but actually x should have shape [N, 1]
            x = u.T @ y_  
            x2 = x ** 2
            # if k < N, we compute sum of xi for 0 singular values directly
            res_x2 = (y_ ** 2).sum() - x2.sum()  

            alpha, beta = 1.0, 1.0
            for _ in range(11):
                t = alpha / beta
                gamma = (sigma / (sigma + t)).sum()
                m2 = (sigma * x2 / ((t + sigma) ** 2)).sum()
                res2 = (x2 / ((1 + sigma / t) ** 2)).sum() + res_x2
                alpha = gamma / (m2 + 1e-5)
                beta = (N - gamma) / (res2 + 1e-5)
                t_ = alpha / beta
                evidence = D / 2.0 * np.log(alpha) \
                           + N / 2.0 * np.log(beta) \
                           - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                           - beta / 2.0 * res2 \
                           - alpha / 2.0 * m2 \
                           - N / 2.0 * np.log(2 * np.pi)
                evidence /= N
                if abs(t_ - t) / t <= 1e-3:  # abs(t_ - t) <= 1e-5 or abs(1 / t_ - 1 / t) <= 1e-5:
                    break
            evidence = D / 2.0 * np.log(alpha) \
                       + N / 2.0 * np.log(beta) \
                       - 0.5 * np.sum(np.log(alpha + beta * sigma)) \
                       - beta / 2.0 * res2 \
                       - alpha / 2.0 * m2 \
                       - N / 2.0 * np.log(2 * np.pi)
            evidence /= N
            m = 1.0 / (t + sigma) * s * x
            m = (vh.T @ m).reshape(-1)
            evidences.append(evidence)
            self.alphas.append(alpha)
            self.betas.append(beta)
            self.ms.append(m)
        self.ms = np.stack(self.ms)
        return np.mean(evidences)

    _fit = _fit_fixed_point
    #_fit = _fit_icml

    def fit(self, f: np.ndarray, y: np.ndarray):
        """
        :param f: [N, F], feature matrix from pre-trained model
        :param y: target labels.
            For classification, y has shape [N] with element in [0, C_t).
            For regression, y has shape [N, C] with C regression-labels

        :return: LogME score (how well f can fit y directly)
        """
        if self.fitted:
            warnings.warn('re-fitting for new data. old parameters cleared.')
            self.reset()
        else:
            self.fitted = True
        f = f.astype(np.float64)
        if self.regression:
            y = y.astype(np.float64)
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
        return self._fit(f, y)

    def predict(self, f: np.ndarray):
        """
        :param f: [N, F], feature matrix
        :return: prediction, return shape [N, X]
        """
        if not self.fitted:
            raise RuntimeError("not fitted, please call fit first")
        f = f.astype(np.float64)
        logits = f @ self.ms.T
        if self.regression:
            return logits
        prob = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  
        # return np.argmax(logits, axis=-1)
        return prob


def LEEP(outputs, y):
    r"""
    Log Expected Empirical Prediction in `LEEP: A New Measure to
    Evaluate Transferability of Learned Representations (ICML 2020)
    <http://proceedings.mlr.press/v119/nguyen20b/nguyen20b.pdf>`_.
    
    The LEEP :math:`\mathcal{T}` can be described as:

    .. math::
        \mathcal{T}=\mathbb{E}\log \left(\sum_{z \in \mathcal{C}_s} \hat{P}\left(y \mid z\right) \theta\left(y \right)_{z}\right)

    where :math:`\theta\left(y\right)_{z}` is the predictions of pre-trained model on source category, :math:`\hat{P}\left(y \mid z\right)` is the empirical conditional distribution estimated by prediction and ground-truth label.

    Args:
        predictions (np.ndarray): predictions of pre-trained model.
        labels (np.ndarray): groud-truth labels.

    Shape: 
        - predictions: (N, :math:`C_s`), with number of samples N and source class number :math:`C_s`.
        - labels: (N, ) elements in [0, :math:`C_t`), with target class number :math:`C_t`.
        - score: scalar
    """
    predictions = F.softmax(outputs, dim=-1).numpy()
    N, C_s = predictions.shape
    labels = y.reshape(-1)
    C_t = int(np.max(labels) + 1)

    normalized_prob = predictions / float(N)
    joint = np.zeros((C_t, C_s), dtype=float)  # placeholder for joint distribution over (y, z)

    for i in range(C_t):
        this_class = normalized_prob[labels == i]
        row = np.sum(this_class, axis=0)
        joint[i] = row

    #p_target_given_source = (joint / joint.sum(axis=0, keepdims=True)).T  # P(y | z)
    #empirical_prediction = predictions @ p_target_given_source
    # Modfied to handle cases where predictions for some y are 0
    j_sum = joint.sum(axis=0, keepdims=True)
    p_target_given_source = np.zeros((joint.shape[0], joint.shape[1]))
    np.divide(joint,  j_sum , out=p_target_given_source , where=j_sum!=0)
    empirical_prediction = predictions @ p_target_given_source.T
    
    empirical_prob = np.array([predict[label] for predict, label in zip(empirical_prediction, labels)])
    score = np.mean(np.log(empirical_prob))

    return score

def NLEEP(X, y, component_ratio=5):

    n = len(y)
    num_classes = len(np.unique(y))
    # PCA: keep 80% energy
    pca_80 = PCA(n_components=0.8, random_state=42)
    pca_80.fit(X, y)
    X_pca_80 = pca_80.transform(X)

    # GMM: n_components = component_ratio * class number
    n_components_num = component_ratio * num_classes
    gmm = GaussianMixture(n_components= n_components_num).fit(X_pca_80)
    prob = gmm.predict_proba(X_pca_80)  # p(z|x)

    # NLEEP
    pyz = np.zeros((num_classes, n_components_num))
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        filter_ = np.take(prob, indices, axis=0) 
        pyz[y_] = np.sum(filter_, axis=0) / n   
    pz = np.sum(pyz, axis=0)    
    py_z = pyz / pz             
    py_x = np.dot(prob, py_z.T) 

    # nleep_score
    nleep_score = np.sum(py_x[np.arange(n), y]) / n

    return nleep_score

def LogME_Score(X, y, regression=False):

    logme = LogME(regression)
    score = logme.fit(X, y)
    return score


def SFDA_Score(X, y):

    n = len(y)
    num_classes = len(np.unique(y))
    
    SFDA_first = SFDA()
    prob = SFDA_first.fit(X, y).predict_proba(X)  # p(y|x)
    
    # soften the probability using softmax for meaningful confidential mixture
    prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True) 
    means, means_ = _class_means(X, y)  # class means, outer classes means
    
    # ConfMix
    for y_ in range(num_classes):
        indices = np.where(y == y_)[0]
        y_prob = np.take(prob, indices, axis=0)
        y_prob = y_prob[:, y_]  # probability of correctly classifying x with label y        
        X[indices] = y_prob.reshape(len(y_prob), 1) * X[indices] + \
                            (1 - y_prob.reshape(len(y_prob), 1)) * means_[y_]
    
    SFDA_second = SFDA(shrinkage=SFDA_first.shrinkage)
    prob = SFDA_second.fit(X, y).predict_proba(X)   # n * num_cls

    # leep = E[p(y|x)]. Note: the log function is ignored in case of instability.
    sfda_score = np.sum(prob[np.arange(n), y]) / n
    return sfda_score


def PARC_Score(X, y, ratio=2):
    
    num_sample, feature_dim = X.shape
    ndims = 32 if ratio > 1 else int(feature_dim * ratio)  # feature reduction dimension

    if num_sample > 15000:
        from utils import initLabeled
        p = 15000.0 / num_sample
        labeled_index = initLabeled(y, p=p)
        features = X[labeled_index]
        targets = X[labeled_index]
        print("data are sampled to {}".format(features.shape))

    method = PARC(n_dims = ndims)
    parc_score = method(features=X, y=y)

    return parc_score

def NCTI_Score(X, y):
    C = np.unique(y).shape[0]
    pca = PCA(n_components=64, random_state=42)
    X = pca.fit_transform(X, y)
    # model_npy_feature = os.path.join('./results_f/group1/pca_feature', f'{args.model}_{args.dataset}_feature.npy')
    # np.save(model_npy_feature, X)
    temp = max(np.exp(-pca.explained_variance_[:32].sum()), 1e-10)
    print(pca.explained_variance_[:32].sum()/ pca.explained_variance_.sum())

    

    if temp == 1e-10:
        clf = LinearDiscriminantAnalysis(solver='svd')

    else:
        clf = LinearDiscriminantAnalysis(solver='eigen', shrinkage=float(temp))
    
    low_feat = clf.fit_transform(X, y)
    
    low_feat = low_feat - np.mean(low_feat, axis=0, keepdims=True)
    all_lowfeat_nuc = np.linalg.norm(low_feat, ord='nuc')

    low_pred = clf.predict_proba(X)
    sfda_score = np.sum(low_pred[np.arange(X.shape[0]), y]) / X.shape[0]
    print(clf.score(X,y))

    class_pred_nuc = 0
    class_low_feat = np.zeros((C, 1))
    print(class_low_feat.shape)
    for c in range(C):
        c_pred = low_pred[(y==c).flatten()]
        c_pred_nuc = np.linalg.norm(c_pred, ord='nuc')
        class_pred_nuc += c_pred_nuc
    print("S_seli: " + str(all_lowfeat_nuc))
    print("S_vc: - " + str((class_pred_nuc)))
    print("S_ncc: " + str((sfda_score)))
    return    all_lowfeat_nuc, sfda_score, np.log(class_pred_nuc)

def LP_Score(X, y):

    n_components = 50
    NCA = NeighborhoodComponentsAnalysis(n_components=n_components, random_state=42)
    NCA.fit(X, y)
    embed_X = NCA.transform(X)
    
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(embed_X, y)
    prob = neigh.predict_proba(embed_X)
    LP_score = np.sum(prob[np.arange(X.shape[0]), y]) / X.shape[0]

    return LP_score

def FU_Score(model, first, second, test_loader, device):
   seed=42
   torch.manual_seed(seed)
   np.random.seed(seed)
   # for cuda
   torch.cuda.manual_seed_all(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   torch.backends.cudnn.enabled = False

   total_loss = []

   model.to(device)
   model.eval()

   # Counter for number of batches
   num_batches = 0
   triplet_loss = losses.TripletMarginLoss()
   conv1_grads = []
   conv2_grads = []
   # Loop over test data
   for batch_idx, (inputs, targets) in enumerate(test_loader):
      inputs, targets = inputs.to(device), torch.squeeze(targets).to(device)

      # Zero the gradients from the previous batch
      model.zero_grad()

      # Forward pass
      embeddings = model(inputs)

      # Compute loss
      loss = triplet_loss(embeddings, targets)
      total_loss.append(loss.item())
      # Backward pass to compute gradients
      loss.backward()

      # Compute L2 norm for the entire 'conv1' layer
      if first.weight.grad is not None:
         conv1_grads.append(torch.norm(first.weight.grad, p=2).item())
      
      # Compute L2 norm for the entire 'layer1' layer
      if second.weight.grad is not None:
         conv2_grads.append(torch.norm(second.weight.grad, p=2).item())

   model.cpu()
   del model
   gc.collect()
   torch.cuda.empty_cache()
   
   return np.mean(conv2_grads) / np.mean(conv1_grads)