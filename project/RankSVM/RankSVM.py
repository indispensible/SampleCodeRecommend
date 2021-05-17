#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
-----------------------------------------
@Author: Lv Gang
@Email: 1547554745@qq.com
@Created: 2021/03/30
------------------------------------------
@Modify: 2021/03/30
------------------------------------------
@Description:
"""

import itertools
import numpy as np

from sklearn import svm, linear_model
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import KFold

"""
Implementation of pairwise ranking using scikit-learn LinearSVC

Reference: "Large Margin Rank Boundaries for Ordinal Regression", R. Herbrich,
    T. Graepel, K. Obermayer.

Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>
"""


def transform_pairwise(X, y, return_list=False):
    """Transforms data into pairs with balanced labels for ranking

    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.

    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1

    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.

    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    if return_list:
        x_new = []
        for item in X_new:
            x_new.append(list(item))
        return x_new, y_new
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(SGDClassifier):
    """Performs pairwise ranking with an underlying LinearSVC model (svm.LinearSVC)

    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.

    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit_with_transform_pairwise(self, x, y):
        print("开始训练模型")
        super(RankSVM, self).fit(x, y)
        print("模型训练结束")
        return self

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)

        Returns
        -------
        self
        """
        print("开始训练模型")
        X_trans, y_trans = transform_pairwise(X, y)
        print("训练数据预处理完毕")
        super(RankSVM, self).fit(X_trans, y_trans)
        print("模型训练结束")
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        The item is given such that items ranked on top have are
        predicted a higher ordering (i.e. 0 means is the last item
        and n_samples would be the item ranked on top).

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            return np.argsort(np.dot(X, self.coef_.T).ravel())
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)


if __name__ == '__main__':
    # as showcase, we will create some non-linear data
    # and print the performance of ranking vs linear regression

    np.random.seed(1)
    n_samples, n_features = 1000, 5
    true_coef = np.random.randn(n_features)
    X = np.random.randn(n_samples, n_features)
    noise = np.random.randn(n_samples) / np.linalg.norm(true_coef)
    y = np.dot(X, true_coef)
    y = np.arctan(y)  # add non-linearities
    y += .1 * noise  # add noise
    Y = y
    # Y = np.c_[y, np.mod(np.arange(n_samples), 5)]  # add query fake id

    train_num = 800

    # make a simple plot out of it
    import pylab as pl

    pl.scatter(np.dot(X, true_coef), y)
    pl.title('Data to be learned')
    pl.xlabel('<X, coef>')
    pl.ylabel('y')
    pl.show()

    # print the performance of ranking
    rank_svm = RankSVM().fit(X[:train_num], Y[:train_num])

    res = rank_svm.predict(X[train_num:])
    new_res = np.c_[Y[train_num:], res]
    new_res = np.sort(new_res)
    a = new_res[np.argsort(new_res[:, 0])]
    b = new_res[np.argsort(new_res[:, 1])]

    print('Performance of ranking ', rank_svm.score(X[train_num:], Y[train_num:]))

    # and that of linear regression
    # ridge = linear_model.RidgeCV(fit_intercept=True)
    # ridge.fit(X[:train_num], Y[:train_num])
    # X_test_trans, y_test_trans = transform_pairwise(X[train_num:], Y[train_num:])
    # score = np.mean(ridge.predict(X_test_trans) == y_test_trans)
    # # score = np.mean(np.sign(np.dot(X_test_trans, ridge.coef_)) == y_test_trans)
    # print('Performance of linear regression ', score)
