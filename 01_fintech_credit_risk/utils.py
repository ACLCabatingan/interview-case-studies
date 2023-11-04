import numpy as np
import pandas as pd
import scipy.stats as ss
import ast 
from sklearn.metrics import precision_recall_curve, auc
from scipy.stats import norm



def get_trunc_corr_df(data, th):
    return data \
        .where(np.triu(np.ones(data.shape),k=1).astype(np.bool)) \
            .where(data.apply(abs) >= th, np.nan) \
                .dropna(axis=0, how='all').dropna(axis=1, how='all')


def pr_auc_score(y_test, y_pred):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    return auc(recall, precision)


# def plotter(df, columns, target_col, nrows, ncols, **kwargs):
#     fig, axs = plt.subplots(nrows, ncols, **kwargs)
#     for i,c in enumerate(columns):
#         ax = axs[int(i/ncols)-1,i%ncols]


# https://stackoverflow.com/questions/25122999/scikit-learn-how-to-check-coefficients-significance
def logit_pvalue(model, x):
    """ Calculate z-scores for scikit-learn LogisticRegression.
    parameters:
        model: fitted sklearn.linear_model.LogisticRegression with intercept and large C
        x:     matrix on which the model was fit
    This function uses asymtptics for maximum likelihood estimates.
    """
    p = model.predict_proba(x)
    n = len(p)
    m = len(model.coef_[0]) + 1
    coefs = np.concatenate([model.intercept_, model.coef_[0]])
    x_full = np.matrix(np.insert(np.array(x), 0, 1, axis = 1))
    ans = np.zeros((m, m))
    for i in range(n):
        ans = ans + np.dot(np.transpose(x_full[i, :]), x_full[i, :]) * p[i,1] * p[i, 0]
    vcov = np.linalg.inv(np.matrix(ans))
    se = np.sqrt(np.diag(vcov))
    t =  coefs/se  
    p = (1 - norm.cdf(abs(t))) * 2
    return p


# https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
def cramers_corrected_stat(confusion_matrix):
    """ calculate Cramer's V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher, 
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r,k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))    
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))


# https://stackoverflow.com/questions/65295837/turn-string-representation-of-interval-into-actual-interval-in-pandas
def interval_type(s):
    """Parse interval string to Interval"""
    
    table = str.maketrans({'[': '(', ']': ')'})
    left_closed = s.startswith('[')
    right_closed = s.endswith(']')

    left, right = ast.literal_eval(s.translate(table))

    t = 'neither'
    if left_closed and right_closed:
        t = 'both'
    elif left_closed:
        t = 'left'
    elif right_closed:
        t = 'right'

    return pd.Interval(left, right, closed=t)