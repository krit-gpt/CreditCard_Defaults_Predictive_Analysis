# -*- coding: utf-8 -*-

## Python Code

## Loading libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
%matplotlib inline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from Class_replace_impute_encode import ReplaceImputeEncode
from Class_regression import logreg
from Class_tree import DecisionTree
from Class_FNN import NeuralNetwork
from sklearn.metrics import roc_curve, auc
from Class_FNN import NeuralNetwork
from sklearn import tree
from sklearn.tree import export_graphviz
from pydotplus import graph_from_dot_data
import graphviz

## Reading data
data= pd.read_excel("CreditCard_Defaults.xlsx")


## defining a function to create barplots
def barplot(column):
    data[column].value_counts().plot(kind='barh')
    plt.xlabel('Frequency')
    plt.ylabel('Value')
    plt.show()
    
barplot('Education')
barplot('Marital_Status')


## Modifying Education variable
low = (data['Education'] == 5) | (data['Education'] == 6) | (data['Education'] == 0)
data.loc[low, 'Education'] = 4
data['Education'].value_counts()

## Modifying Marital_Status variable
data.loc[data['Marital_Status']== 0, 'Marital_Status'] = 3
data['Marital_Status'].value_counts()


## Creating Bins in Age variable
data['AgeBin'] = 0 #creates a column of 0
data.loc[((data['Age'] > 20) & (data['Age'] < 30)) , 'AgeBin'] = 1
data.loc[((data['Age'] >= 30) & (data['Age'] < 40)) , 'AgeBin'] = 2
data.loc[((data['Age'] >= 40) & (data['Age'] < 50)) , 'AgeBin'] = 3
data.loc[((data['Age'] >= 50) & (data['Age'] < 60)) , 'AgeBin'] = 4
data.loc[((data['Age'] >= 60) & (data['Age'] < 70)) , 'AgeBin'] = 5
data.loc[((data['Age'] >= 70) & (data['Age'] < 81)) , 'AgeBin'] = 6


## Dropping Age variable
data.drop('Age', axis=1, inplace=True)

sns.distplot(data['Credit_Limit'], bins=15);
sns.distplot(np.log(data['Credit_Limit']+1), bins=15);  ## After applying Log transformation


## Creating a function to create correlation heatmaps
def CorrelationPlot (df):
    colormap = plt.cm.RdBu
    plt.figure(figsize=(20,20))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df.corr(),linewidths=0.1,vmax=1.0, 
                square=True, cmap = "coolwarm", fmt = ".2f",linecolor='white', annot=True)

attribute_map = {
    'Default':[1,(0,1),[0,0]],
    'Gender':[1,(1,2),[0,0]],
    'Education':[2,(1,2,3,4),[0,0]],
    'Marital_Status':[2,(1,2,3),[0,0]],
    'card_class':[2,(1,2,3),[0,0]],
    'AgeBin':[2,(1,2,3,4,5,6),[0,0]],
    'Credit_Limit':[0,(100,80000),[0,0]],
    'Jun_Status':[0,(-2,8),[0,0]],
    'May_Status':[0,(-2,8),[0,0]],
    'Apr_Status':[0,(-2,8),[0,0]],
    'Mar_Status':[0,(-2,8),[0,0]],
    'Feb_Status':[0,(-2,8),[0,0]],
    'Jan_Status':[0,(-2,8),[0,0]],
    'Jun_Bill':[0,(-12000,32000),[0,0]],
    'May_Bill':[0,(-12000,32000),[0,0]],
    'Apr_Bill':[0,(-12000,32000),[0,0]],
    'Mar_Bill':[0,(-12000,32000),[0,0]], 
    'Feb_Bill':[0,(-12000,32000),[0,0]],
    'Jan_Bill':[0,(-12000,32000),[0,0]],
    'Jun_Payment':[0,(0,60000),[0,0]],
    'May_Payment':[0,(0,60000),[0,0]],
    'Apr_Payment':[0,(0,60000),[0,0]],
    'Mar_Payment':[0,(0,60000),[0,0]],
    'Feb_Payment':[0,(0,60000),[0,0]],
    'Jan_Payment':[0,(0,60000),[0,0]],
    'Jun_PayPercent':[0,(0,1),[0,0]],
    'May_PayPercent':[0,(0,1),[0,0]],
    'Apr_PayPercent':[0,(0,1),[0,0]],
    'Mar_PayPercent':[0,(0,1),[0,0]],
    'Feb_PayPercent':[0,(0,1),[0,0]],
    'Jan_PayPercent':[0,(0,1),[0,0]]}


rie = ReplaceImputeEncode(data_map=attribute_map, display=True,
                          nominal_encoding='one-hot', drop=False)
encoded_df = rie.fit_transform(data)

encoded_df['Credit_Limit']=np.log(encoded_df['Credit_Limit']+1)

CorrelationPlot(encoded_df)




encoded_df1= encoded_df.drop(['Jun_Bill', 'May_Bill', 'Apr_Bill', 'Mar_Bill', 
                              'Feb_Bill', 'Jan_Bill','May_Status','Apr_Status',
                              'Mar_Status','Feb_Status','Jan_Status',
                              'card_class0', 'card_class2', 'Marital_Status0', 
                              'Education0', 'AgeBin0'], axis=1).copy()


CorrelationPlot(encoded_df1)
plt.title('Pearson Correaltion of Features from final DataFrame')
plt.show()


y = np.asarray(encoded_df1['Default']) 
# Drop the target, 'object'.  Axis=1 indicates the drop is for a column.
X = np.asarray(encoded_df1.drop('Default', axis=1)) 

##Splitting data
X_train, X_validate, y_train, y_validate = train_test_split(X,y,test_size = 0.3, random_state=7)
score_list = ['accuracy', 'recall', 'precision', 'f1']
 
## Cross-validation on Logistic Regression
class_weight=['None', 'balanced']
max_f1 = 0
for i in class_weight:
    print("Class Weight: ", i)
    lgr = LogisticRegression(class_weight=i, solver='newton-cg', max_iter=10000)

    logreg_scores = cross_validate(lgr, X, np.ravel(y), scoring=score_list, \
                                        return_train_score=False, cv=10)
    for s in score_list:
        var = "test_"+s
        mean = logreg_scores[var].mean()
        std  = logreg_scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    if mean > max_f1:
        max_f1 = mean
        best_class_weight    = i
print("Best class weight: " ,best_class_weight)

lgr_train = LogisticRegression(class_weight='balanced',solver='newton-cg',
                               max_iter=10000).fit(X_train,y_train)

## Validating on the test data
print("\nTraining Data\nRandom Selection of 70% of Original Data")
logreg.display_binary_split_metrics(lgr_train, X_train, y_train, \
                                    X_validate, y_validate)

varlist = ['Default']
X = encoded_df1.drop(varlist, axis=1).copy()
y = encoded_df1[varlist]
np_y = np.ravel(y)  

## Cross-validation on Random Forest
estimators_list   = [10, 15, 20]
max_features_list = ['auto', 0.3, 0.5, 0.7]
score_list = ['accuracy', 'recall', 'precision', 'f1']
max_f1 = 0
for e in estimators_list:
    for f in max_features_list:
        print("\nNumber of Trees: ", e, " Max_features: ", f)
        rfc = RandomForestClassifier(n_estimators=e, criterion="gini", \
                    max_depth=None, min_samples_split=2, \
                            min_samples_leaf=1, max_features=f, \
                            n_jobs=1, bootstrap=True, random_state=12345, class_weight='balanced')
        rfc= rfc.fit(X, np_y)
        scores = cross_validate(rfc, X, np_y, scoring=score_list, \
                                        return_train_score=False, cv=10)

        print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
        for s in score_list:
            var = "test_"+s
            mean = scores[var].mean()
            std  = scores[var].std()
            print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
        if mean > max_f1:
            max_f1 = mean
            best_estimator    = e
            best_max_features = f

print("\nBest based on F1-Score")
print("Best Number of Estimators (trees) = ", best_estimator)
print("Best Maximum Features = ", best_max_features)

rfc_train = RandomForestClassifier(n_estimators=best_estimator, criterion="gini", \
                    max_depth=10, min_samples_split=2, \
                    min_samples_leaf=1, max_features= best_max_features,\
                    n_jobs=1, bootstrap=True, random_state=12345, 
                    class_weight='balanced').fit(X_train, y_train)

## Validating on the test data
print("\nTraining Data\nRandom Selection of 70% of Original Data")
DecisionTree.display_binary_split_metrics(rfc_train, X_train, y_train, \
                                              X_validate, y_validate)
DecisionTree.display_importance(rfc, encoded_df1.columns)


## Cross validation on Decision Trees
max_depth=[5,6,7,8,10,12,15,20,25]
for i in max_depth:
    dtc = DecisionTreeClassifier(criterion='gini', max_depth=i, \
    min_samples_split=5, min_samples_leaf=5)
    dtc = dtc.fit(X,y)
    score_list = ['accuracy', 'recall', 'precision', 'f1']
    mean_score = []
    std_score = []
    print("For max_depth=",i)
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        dtc_10 = cross_val_score(dtc, X, y, scoring=s, cv=10)
        mean = dtc_10.mean()
        std = dtc_10.std()
        mean_score.append(mean)
        std_score.append(std)
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
print("Max depth for the Decision Tree is 6")

dtc_train = DecisionTreeClassifier(criterion='gini', max_depth=6, \
                                   min_samples_split=5, min_samples_leaf=5, 
                                   class_weight='balanced').fit(X_train, y_train)

print("\nTable of the metrics for 70/30 split")
DecisionTree.display_binary_split_metrics(dtc_train, X_train,y_train,X_validate, y_validate)


def dtc_graph():
    dot_data = tree.export_graphviz(dtc_train, out_file=None,
    feature_names=list(X.columns),
    class_names=['0','1'],
    filled=True, rounded=True,
    special_characters=True)
    graph = graphviz.Source(dot_data)
    return(graph)
    
dtc_graph()



## Cross-validation on Neural Networks
network_list = [(3), (11), (5,4), (6,5), (7,6)]
# Scoring for Interval Prediction Neural Networks
max_f1_fnn=0.0
for nn in network_list:
    print("\nNetwork: ", nn)
    fnn = MLPClassifier(hidden_layer_sizes=nn, activation='logistic', \
                        solver='lbfgs', max_iter=1000, random_state=12345)
    fnn = fnn.fit(X,np_y)
    score_list = ['accuracy', 'recall', 'precision', 'f1']
    mean_score = []
    std_score  = []
    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    scores = cross_validate(fnn, X, np_y, scoring=score_list, \
                                    return_train_score=False, cv=10)

    print("{:.<13s}{:>6s}{:>13s}".format("Metric", "Mean", "Std. Dev."))
    for s in score_list:
        var = "test_"+s
        mean = scores[var].mean()
        std  = scores[var].std()
        print("{:.<13s}{:>7.4f}{:>10.4f}".format(s, mean, std))
    if mean > max_f1_fnn:
        max_f1_fnn = mean
        best_nn=nn
print('Best neural network configuration is :',best_nn)            


## Validating on test data
bestfnn = MLPClassifier(hidden_layer_sizes=best_nn, activation='logistic', \
                    solver='lbfgs', max_iter=1000, random_state=12345).fit(X_train, y_train)

NeuralNetwork.display_metrics(bestfnn, X_train, y_train,\
                             X_validate, y_validate)



def roc_curves():
    fpr, tpr, _ = roc_curve(y_validate, rfc_train.predict(X_validate))
    AUC  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Random Forest (AUC = %0.4f)' % AUC)
    fpr, tpr, _ = roc_curve(y_validate, lgr_train.predict(X_validate))
    AUC  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Logistic Regression (AUC = %0.4f)' % AUC)
    fpr, tpr, _ = roc_curve(y_validate, dtc_train.predict(X_validate))
    AUC  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Decision Tree (AUC = %0.4f)' % AUC)
    fpr, tpr, _ = roc_curve(y_validate, bestfnn.predict(X_validate))
    AUC  = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='Neural Network (AUC = %0.4f)' % AUC)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    
roc_curves()











