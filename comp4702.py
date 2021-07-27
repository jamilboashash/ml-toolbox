"""
This file contains some functions written throughout the duration of the
 COMP4702 course. They have been organised and prepared in this manner for
 use in the course final exam.

Author: Jamil Boashash
"""

# import libraries
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, plot_confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder

import numpy as np
import operator
from sklearn.tree import DecisionTreeClassifier


def to_numeric(df, col: str):
    """
    todo
    :param df:
    :param col:
    :return: Returns the whole df with the given col converted to a numeric dtype
    """
    label_encoder = LabelEncoder()
    label_encoder.fit(df[col])
    df[col] = label_encoder.transform(df[col])
    return df


def plot_corr_matrix(df):
    """
    Given a dataframe, plot the matrix of Pearson correlation coefficients.
    :param df: pandas dataframe
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(df.corr(), cmap='Blues', ax=ax)


def evaluate(model, X_train, y_train, X_test, y_test):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    axes[0].title.set_text('Training Set')
    plot_confusion_matrix(model, X_train, y_train, ax=axes[0], cmap=plt.cm.Blues)

    axes[1].title.set_text('Test Set')
    plot_confusion_matrix(model, X_test, y_test, ax=axes[1], cmap=plt.cm.Blues)


def conf_matrix(y_t, y_pred):
    """
    todo - test this function
    :param y_t:
    :param y_pred:
    :return:
    """
    confusion = confusion_matrix(y_t, y_pred)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]

    return TP, TN, FP, FN


def scores(y_train, y_pred_train, y_test, y_pred_test):
    print("TRAINING SET RESULTS:")
    print("---------------------")
    precision, recall, fbeta, support = precision_recall_fscore_support(y_train, y_pred_train)
    acc_percent = accuracy_score(y_true=y_train, y_pred=y_pred_train, normalize=True)
    acc_samples = accuracy_score(y_true=y_train, y_pred=y_pred_train, normalize=False)

    print(f"Overall Accuracy:           {round(acc_percent, 3) * 100}%")
    print(f"Overall Accuracy (#):       {acc_samples} / {y_train.shape[0]}")
    print(f"Classification Error:       {round(1 - acc_percent, 3) * 100}%")
    print(f"Classification Error (#):   {y_train.shape[0] - acc_samples} / {y_train.shape[0]}")
    print(f'Precision (classA):         {round(precision[0], 3) * 100}%')
    print(f'Precision (classB):         {round(precision[1], 3) * 100}%')
    print(f'Recall (classA):            {round(recall[0], 3) * 100}%')
    print(f'Recall (classB):            {round(recall[1], 3) * 100}%')
    print(f'F-beta (classA):            {round(fbeta[0], 3) * 100}%')
    print(f'F-beta (classB):            {round(fbeta[1], 3) * 100}%')
    print(f'Support (classA):           {support[0]}')
    print(f'Support (classB):           {support[1]}\n')

    print("TEST SET RESULTS:")
    print("-----------------")
    precision, recall, fbeta, support = precision_recall_fscore_support(y_test, y_pred_test)
    acc_percent = accuracy_score(y_true=y_test, y_pred=y_pred_test, normalize=True)
    acc_samples = accuracy_score(y_true=y_test, y_pred=y_pred_test, normalize=False)

    print(f"Overall Accuracy:           {round(acc_percent, 3) * 100}%")
    print(f"Overall Accuracy (#):       {acc_samples} / {y_test.shape[0]}")
    print(f"Classification Error:       {round(1 - acc_percent, 3) * 100}%")
    print(f"Classification Error (#):   {y_test.shape[0] - acc_samples} / {y_test.shape[0]}")
    print(f'Precision (classA):         {round(precision[0], 3) * 100}%')
    print(f'Precision (classB):         {round(precision[1], 3) * 100}%')
    print(f'Recall (classA):            {round(recall[0], 3) * 100}%')
    print(f'Recall (classB):            {round(recall[1], 3) * 100}%')
    print(f'F-beta (classA):            {round(fbeta[0], 3) * 100}%')
    print(f'F-beta (classB):            {round(fbeta[1], 3) * 100}%')
    print(f'Support (classA):           {support[0]}')
    print(f'Support (classB):           {support[1]}\n')


def target_input_split(df, target: str):
    """
    Given a dataframe (df) and the target feature (target), split the df into
     input and target features.
    :param df: DataFrame
    :param target: target feature column name (as a string)
    :return: Returns (1) the input features, (2) the target feature
    """
    input_features = df.columns.values.tolist()
    input_features.remove(target)
    return df[input_features], df[target]


def plot_tsne(df_2d, col_2d, df_3d, col_3d):
    fig = plt.figure(figsize=(20, 15))

    # 2d plot
    fig.add_subplot(221)
    plt.scatter(df_2d[:, 0], df_2d[:, 1], c=col_2d, alpha=0.5)

    # 3d plot
    fig.add_subplot(222, projection='3d')
    plt.scatter(df_3d[:, 0], df_3d[:, 1], df_3d[:, 2], c=col_3d, alpha=0.5)

    plt.show()


def n_degree_polynomial(X, y, degree=2):
    poly_feat = PolynomialFeatures(degree)
    x_poly = poly_feat.fit_transform(X)

    model = LinearRegression().fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)

    print("Model results:")
    print("-----------------------------------")
    print(f"model coefficients = {model.coef_}")
    print(f"model intercept    = {model.intercept_}")
    print(f"model RMSE         = {np.sqrt(mean_squared_error(y, y_poly_pred))}")  # we want to minimise this
    print(f"model R2 score     = {r2_score(y, y_poly_pred)}")  # we want to maximise this
    print("-----------------------------------")

    # plot the regression line over the data (scatter plot)
    plt.figure(figsize=(15, 7))
    plt.scatter(X, y)

    # sort the X values before the line plot
    sort_axis = operator.itemgetter(0)
    sorted_zip = sorted(zip(X, y_poly_pred), key=sort_axis)
    X_sorted, y_poly_pred = zip(*sorted_zip)

    plt.plot(X_sorted, y_poly_pred, 'r')


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score


def eval_poly_reg_rmse(X_train, y_train, from_m, to_n):
    # lists to collect scores
    train_scores = list()
    test_scores = list()

    # range of tree depths to evaluate
    values = [degree for degree in range(from_m, to_n)]

    # model & evaluate a tree for each depth
    for degree in values:
        poly_feat = PolynomialFeatures(degree)
        x_poly = poly_feat.fit_transform(X_train)

        model = LinearRegression().fit(x_poly, y_train)

        # evaluate
        y_pred_train = model.predict(x_poly)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        train_scores.append(rmse)

    # plot of accuracy vs tree depth
    plt.plot(values, train_scores, '-o', label='Train')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()


def eval_poly_reg_r2score(X_train, y_train, from_m, to_n):
    # lists to collect scores
    train_scores = list()
    test_scores = list()

    # range of tree depths to evaluate
    values = [degree for degree in range(from_m, to_n)]

    # model & evaluate a tree for each depth
    for degree in values:
        poly_feat = PolynomialFeatures(degree)
        x_poly = poly_feat.fit_transform(X_train)

        model = LinearRegression().fit(x_poly, y_train)

        # evaluate
        y_pred_train = model.predict(x_poly)
        r2 = r2_score(y_train, y_pred_train)
        train_scores.append(r2)

    # plot of accuracy vs tree depth
    plt.plot(values, train_scores, '-o', label='Train')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('R2 Score')
    plt.legend()
    plt.show()


def eval_depth_vs_accuracy(X_train, y_train, X_test, y_test):
    from sklearn.metrics import accuracy_score

    # lists to collect scores
    train_scores = list()
    test_scores = list()

    # range of tree depths to evaluate
    values = [i for i in range(1, 16)]

    # model & evaluate a tree for each depth
    for i in values:

        model = DecisionTreeClassifier(criterion='gini', splitter='random', max_depth=i, random_state=0)
        model.fit(X_train, y_train)

        # evaluate on training set
        y_pred_train_dt = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_pred_train_dt)
        train_scores.append(train_acc)

        # evaluate on test set
        y_pred_test_dt = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test_dt)
        test_scores.append(test_acc)

    # plot of accuracy vs tree depth
    plt.plot(values, train_scores, '-o', label='Train')
    plt.plot(values, test_scores, '-o', label='Test')
    plt.xlabel('tree depth')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
