# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# read in data
df = pd.read_csv("Social_Network_Ads.csv")

# Print first 5 rows of the dataset
print(df.head())

# Check for null values and what type our columns are
df.info()

# First we look at target variable porportions
pd.crosstab(df['Purchased'], df['Purchased'], normalize='all')*100

df.isnull().sum()

# drop not used columns
df.drop('User ID', axis=1, inplace=True)

df['Gender']=np.where(df['Gender']=='Male',1,0)

print(df.columns)

# Rearange column order
df = df[['Purchased', 'Age', 'EstimatedSalary', 'Gender']]

df.head()

# useing heatmap to see the correlation between our features
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.show()

sns.set_style('whitegrid')
sns.pairplot(df,hue='Purchased')
plt.show()


y = df['Purchased'] # dependent features
X = df.drop('Purchased', axis=1) # independent features

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1000)

# Feature Scaling: only for visualization purposes we need to scale the features. 
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test) 

# Import Machine Learning Model DecissionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

#Post proning
# Set the DecisionTreeClassifier in a class
Tree_classifier = DecisionTreeClassifier(max_depth=4)

# Fitting our train data
Tree_classifier.fit(X_train, y_train)

# Predicting the Test set classes
y_hat = Tree_classifier.predict(X_test) #y_hat are our predictions

# from sklearn import tree

# plt.figure(figsize=(12,10))
# tree.plot_tree(Tree_classifier,filled=True)
# plt.show()

#preproning
clfr = DecisionTreeClassifier()

parameters = {"criterion": ("gini","entropy","log_loss"), "max_depth": [1,2,3,4,5,6,7,8,9,10,11]}

#TO find the best parameter
from sklearn.model_selection import GridSearchCV
gsv = GridSearchCV(clfr,parameters,scoring = 'accuracy', cv = 5)

gsv.fit (X_train,y_train)
print(gsv.best_params_)
print((gsv.best_score_)*100)


from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,confusion_matrix
print(confusion_matrix(y_test, y_hat))
print(accuracy_score(y_test, y_hat))
print(recall_score(y_test, y_hat))
print(precision_score(y_test, y_hat))
print(f1_score(y_test, y_hat))

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

rfc = RandomForestClassifier()
ss = SVC()
Dtc = DecisionTreeClassifier()
knn = KNeighborsClassifier()

algo = [ rfc,ss, Dtc, knn]
algo_name = ["RandomForestClassifier", "SVC","DecisionTreeClassifier","KNeighborsClassifier"]

from sklearn.metrics import accuracy_score,log_loss

for i,j in zip (algo,algo_name):
  i.fit(X_train,y_train)
  pred=i.predict(X_test)
  print(j,'\n')
  print('Accuracy score:{:.2f}%'.format(accuracy_score(y_test,pred)*100))
  print("*"*40)

