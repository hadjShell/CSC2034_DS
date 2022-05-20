"""
Generally helpful functions. 
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn
from collections import Counter
from sklearn.metrics import roc_curve, auc, roc_auc_score
from matplotlib.colors import ListedColormap


def show_scatterplot(data, labels, title, xlabel = 'x values', ylabel = 'y values', figsize = (10, 6)):
    """
    Produce scatterplot for passed data.
    
    Arguments:
        data: The data to plot
            Shape: (x, y)
        labels: Class labels for data
            Shape: (x, )
        title: Plot title
            Type: String
        xlabel (optional): x axis label
            Type: String
        ylabel (optional): y axis label
            Type: String
        figsize (optional): Figure size (inches)
            Type: (int, int)
    Returns:
        None
        
    
    Author: Cameron Trotter
    Email: c.trotter2@ncl.ac.uk
    """
    plt.figure(figsize = figsize)
    sns.scatterplot(x = data[:,0], y = data[:,1], hue = labels)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.close()
    
    
def plot_linear_fit(clf, data, labels, title):
    """
    Produce visualisation of linear fit.
    
    Arguments:
        clf: The classifier trained
            Type: sklearn linear model
        data: The data to plot
            Shape: (x, y)
        labels: Class labels for data
            Shape: (x, )
        title: Plot title
            Type: String
    Returns:
        None
        
    
    Author: Paolo Missier
    Email: paolo.missier@ncl.ac.uk
    
    """
    
    data_min, data_max = data[:, 0].min() - .5, data[:, 0].max() + .5
    
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(data_min, data_max)
    yy = a * xx - (clf.intercept_[0]) / w[1]

    fig = plt.figure(figsize=(20,6))
    fig.subplots_adjust(hspace=1, wspace=0.4)

    ax = fig.add_subplot(1, 2, 1)
    sns.scatterplot(x=data[:,0],y=data[:,1], hue=labels)
    ax.set_title(title)
    ax2 = ax.twinx()
    sns.regplot(x=xx,y=yy, ax=ax2, line_kws={"color": "red"}, scatter = False)

    plt.show()
    plt.close()
    
    
def plot_confusion_matrix(conf_matrix, labels, title):
    """
    Plot confusion matrix
    
    Arguments:
        conf_matrix: A produced confusion matrix
        labels: Class labels for data
            Shape: (x, )
        title: Plot title
            Type: String
    Returns:
        None
        
    Author: Cameron Trotter
    Email: c.trotter2@ncl.ac.uk
    """
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.show(block=False)
    
def plot_ROC(clf, XTest, CLTest, CL_pred_Test=None):
    """
    Plot ROC
    
    Arguments:
        clf: The classifier trained
            Type: sklearn model
        XTest: Test set data points
            Shape: (x, y)
        CLTest: Class labels per data point
            Shape: (x, )
        CL_pred_Test (optional): Predictions on test set
            Shape: (x, )
            Default: None
            
    Author: Paolo Missier
    Email: paolo.missier@ncl.ac.uk       
    
    """
    
    if hasattr(clf, "decision_function"):
        print("using decision_function")
        probs = clf.decision_function(XTest)
        preds = probs
    else:
        print("using predict_proba")
        probs = clf.predict_proba(XTest)
        preds = probs[:,1]

    fpr, tpr, threshold = roc_curve(CLTest, preds)
    roc_auc = auc(fpr, tpr)

    if CL_pred_Test is not None:
        print("\n\n====== ROC ======")
        print("roc_auc_score = %0.2f" % roc_auc_score(CLTest, CL_pred_Test))
        print("auc = %0.2f" % roc_auc)

    fig = plt.figure()
    plt.title('ROC curve')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

def plot_contour_fit(clf, XTrain, CLTrain, XTest, CLTest):
    """
    Plot contour fit
    
    Arguments:
        clf: The classifier trained
            Type: sklearn model
        XTrain: Train set data points
            Shape: (x, y)
        CLTrain: Train class labels per data point
            Shape: (x, )
        XTest: Test set data points
            Shape: (x, y)
        CLTest: Test class labels per data point
            Shape: (x, )
            
    Author: Paolo Missier
    Email: paolo.missier@ncl.ac.uk       
    
    """

    h = .02  # step size in the mesh

    fig = plt.figure(figsize=(20,6))
    fig.subplots_adjust(hspace=1, wspace=0.4)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])


    ## plot training set
    x_min, x_max = XTrain[:, 0].min() - .5, XTrain[:, 0].max() + .5
    y_min, y_max = XTrain[:, 1].min() - .5, XTrain[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax = fig.add_subplot(1, 2, 1)
    sns.scatterplot(x=XTrain[:,0],y=XTrain[:,1], hue=CLTrain, ax=ax)  # plot training set
    ax.set_title("training set wih contour line")
    
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
    
    ## plot test set
    x_min, x_max = XTest[:, 0].min() - .5, XTest[:, 0].max() + .5
    y_min, y_max = XTest[:, 1].min() - .5, XTest[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    ax = fig.add_subplot(1, 2, 2)
    sns.scatterplot(x=XTest[:,0],y=XTest[:,1], hue=CLTest, ax=ax)  # plot training set
    ax.set_title("test set wih contour line")
    
    # Plot the decision boundary.
    if hasattr(clf, "decision_function"):
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    else:
        Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
    
def create_imbalance(labels, percentage_imbalance):
    """
    Given some labels for a dataset, randomly flip percentage_inbalance of them.
    
    Arguments:
        labels: Dataset labels
            Shape: (x, )
        percentage_inbalance: percentage of labels to flip
            Type: float (0.0 - 1.0)
    Returns: Dataset labels with percentage_inbalance flipped
        Shape: (x, )
        
    Author: Cameron Trotter & Paolo Missier
    Email: c.trotter2@ncl.ac.uk & paolo.missier@ncl.ac.uk
    
    """
    n = 0
    for i in range(len(labels)):
        if labels[i] == 0:
            if randn() <= percentage_imbalance:
                n += 1
                labels[i] = 1
    print(f"{n} labels values flipped")
    print("class labels ratio: %0.2f" % (Counter(labels)[0] / Counter(labels)[1]))
    return labels
    
def downsample(X,CL):
    """
    Utilise downsampling to balance a dataset.
    
    Arguments:
        X: Data points
            Shape: (x, y)
        CL: Class labels
            Shape: (x, )
    Returns:
        A rebalanced dataset.
    
    Author: Paolo Missier
    Email: paolo.missier@ncl.ac.uk
    """
    
    ## we want to achieve roughly 50% contribution for each class
    currentRatio = Counter(CL)[0] / Counter(CL)[1]
    print("current class labels ratio: %0.2f" % currentRatio)

    if currentRatio < 1:
        majority = 1
        threshold = 1- currentRatio
    else:
        majority = 0
        threshold = 1 - 1/ currentRatio
    
    n = 0
    X_reb = np.arange(0).reshape(0, X.shape[1])
    CL_reb = np.arange(0)
    for i in range(len(CL)):
        if CL[i] == majority and randn() <= threshold:
            ## removing record
            n +=1
        else:
            ## copying record
            CL_reb = np.append(CL_reb, CL[i])
            X_reb = np.append(X_reb, X[i:1+i],  axis=0)

    print("%d majority class records removed "% n)
    print("new class labels ratio: %0.2f" % (Counter(CL_reb)[0] / Counter(CL_reb)[1]))
    print("counts: ",Counter(CL_reb))
    
    return X_reb, CL_reb
    
def discreteAlcohol(df):
    df["alcohol_cat"] = 'L'
    mean = df["alcohol"].mean()
    stddev = df["alcohol"].std()
    for index, series in df.iterrows():
        alcohol = series["alcohol"]
        if alcohol < (mean-stddev):
            df.loc[index, "alcohol_cat"] = 'L'
        elif alcohol > (mean+stddev):
            df.loc[index, "alcohol_cat"] = 'H'
        else:
            df.loc[index, "alcohol_cat"] = 'M'

def discreteSugar(df):
    df["isSweet"] = 0
    threshold = df["residual sugar"].median()

    for index, series in df.iterrows():
        sugar = series["residual sugar"]
        if sugar < threshold:
            df.loc[index, "isSweet"] = 0
        else:
            df.loc[index, "isSweet"] = 1
