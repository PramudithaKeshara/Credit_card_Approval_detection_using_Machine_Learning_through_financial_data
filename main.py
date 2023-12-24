#!/usr/bin/env python
# coding: utf-8

# # **Credit Card Approval Detection**
# #### **Steps:**
# 1. Data Collection
# 2. Data Preparing
# 3. Data Preprocessing
# 4. Exploratory Data Analysis
# 5. Data Transformation
# 6. Model Building
# 7. Model Evaluation

# In[1]:


# Install Requirements:
# !pip install -r requirements.txt


# In[2]:


# Importing Required Libraries for the Project

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, plot_confusion_matrix, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from matplotlib.cm import rainbow
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
from matplotlib import rcParams


# In[3]:


import imblearn
print(imblearn.__version__)


# ## **Data Collection and Description**

# In[4]:


train_df = pd.read_csv("Training Data.csv")


# In[5]:


train_df.head()


# In[6]:


test_df = pd.read_csv("Test Data.csv")


# In[7]:


test_df.head()


# In[8]:


train_df.info()


# In[9]:


train_df.describe()


# In[10]:


train_df.isnull().sum()


# In[11]:


train_df.describe(include='object')


# ## **Exploratory Data Analysis**
# Extracting Insights using Aesthetic Visualizations

# In[13]:


train_df.approval.value_counts().plot(kind="bar", color=["salmon", "lightblue"])
plt.xlabel('Approval')
plt.ylabel('counts')
plt.title('Histogram of Predicted Credit Card Approval')


# In[14]:


categorical_val = []
continous_val = []
for column in train_df.columns:
    print('==============================')
    print(f"{column} : {train_df[column].unique()}")
    if len(train_df[column].unique()) <= 10:
        categorical_val.append(column)
    else:
        continous_val.append(column)


# In[15]:


plt.figure(figsize=(15, 15))

for i, column in enumerate(categorical_val, 1):
    plt.subplot(3, 3, i)
    train_df[train_df["approval"] == 0][column].hist(bins=35, color='blue', label='Approval = NO', alpha=0.6)
    train_df[train_df["approval"] == 1][column].hist(bins=35, color='red', label='Approval = YES', alpha=0.6)
    plt.legend()
    plt.xlabel(column)


# In[17]:


train_df.corrwith(train_df.approval).plot(kind='bar', grid=True, figsize=(12, 8), 
                                                   title="Correlation with target column")


# In[18]:


plt.figure(figsize=(18,12))
corr = train_df.corr()
sns.heatmap(corr,annot=True,cmap = plt.cm.cividis)
plt.show()


# In[20]:


plt.figure(figsize = (12,10))
sns.pairplot(train_df,hue='approval',palette='husl')
plt.show()


# ## **Data Transformation** (Categorical - Numerical)

# In[21]:


train_df.info()


# In[22]:


test_df.info()


# In[23]:


train_df.head()


# In[24]:


test_df.head()


# In[25]:


train_df['married'].unique()


# In[26]:


mapping = {'single':'0', 'married':'1'}
train_df['married'] = train_df['married'].map(mapping)


# In[27]:


test_df['married'] = test_df['married'].map(mapping)


# In[ ]:





# In[28]:


train_df['house_ownership'].unique()


# In[29]:


mapping = {'owned':'2','rented':'1','norent_noown':'0'}
train_df['house_ownership'] = train_df['house_ownership'].map(mapping)


# In[30]:


test_df['house_ownership'] = test_df['house_ownership'].map(mapping)


# In[ ]:





# In[31]:


train_df['car_ownership'].unique()


# In[32]:


train_df['car_ownership'] = train_df['car_ownership'].replace({'no':'0', 'yes':'1'})


# In[33]:


test_df['car_ownership'] = test_df['car_ownership'].replace({'no':'0', 'yes':'1'})


# In[ ]:





# ##### **Performing One Hot Encoding on Column Professsion.**

# In[34]:


train_df['profession'].value_counts().head(30)


# In[35]:


top_30_train = [x for x in train_df['profession'].value_counts().sort_values(ascending=False).head(30).index]


# In[36]:


for label in top_30_train:
    train_df[label] = np.where(train_df['profession'] == label, 1,0)
train_df[['profession']+top_30_train].head(10)


# In[ ]:





# In[37]:


top_30_test = [x for x in test_df['profession'].value_counts().sort_values(ascending=False).head(30).index] 


# In[38]:


for label in top_30_test:
    test_df[label] = np.where(test_df['profession'] == label, 1,0)
test_df[['profession']+top_30_test].head(10)


# In[ ]:





# In[39]:


train_df['city'].nunique()


# In[40]:


train_df['city'].value_counts().head(30)


# In[41]:


# Trasforming Variable City into Numerical Values taking only the most risky cities which are more prone to credit card fradualent.
# The reason for taking only top 30 cities are, since performing one hot encoding to
# all the cities can lead to curse of dimenrsionality and it will increase the time complexity.

top_30 = [x for x in train_df['city'].value_counts().sort_values(ascending = False).head(30).index]


# In[42]:


for label in top_30:
    train_df[label] = np.where(train_df['city'] == label ,1,0)
train_df[['city']+top_30].head(50)


# In[ ]:





# In[43]:


top_30_test = [x for x in test_df['city'].value_counts().sort_values(ascending = False).head(30).index]


# In[44]:


for label in top_30_test:
    test_df[label] = np.where(test_df['city'] == label ,1,0)
test_df[['city']+top_30_test].head(50)


# In[ ]:





# In[45]:


train_df['state'].nunique()


# In[46]:


train_df['state'].value_counts()


# In[47]:


# In the case of State we have direct access the state codes in order to convert the column.

train_df['state'] = train_df['state'].astype('category')
train_df['state'] = train_df['state'].cat.codes
train_df['state']


# In[ ]:





# In[48]:


test_df['state'] = test_df['state'].astype('category')
test_df['state'] = test_df['state'].cat.codes
test_df['state']


# In[49]:


train_df.info()


# In[50]:


test_df.info()


# In[51]:


# Dropping Unnecessary Columns

columns=['Id','city','profession']
train = train_df.drop(columns=columns, axis = 1)


# In[52]:


columns=['id','city','profession']
test = test_df.drop(columns=columns)


# In[ ]:





# ### **Seperating the Variables..**

# In[53]:


target_var = train['approval']
train_var = train.drop(['approval'],axis=1)
test_set = test


# ### **Resampling:** 
# - The Reason for resampling our data is to make sure that the data is balanced.
# - We will use the imblearn library to perform the resampling.
# - since at the start we have seen the target variable was imbalanced, where the Approval class 1 feature rows were high in number than 0.
# 
# 

# In[54]:


from imblearn.over_sampling import SMOTE
x_resample,y_resample = SMOTE().fit_resample(train_var,target_var.values.ravel())
print(x_resample.shape)
print(y_resample.shape)


# In[55]:


print("Target_var Before Resampling:")
print(target_var.value_counts())
print("Target_var After Resampling:")
y_resample = pd.DataFrame(y_resample)
print(y_resample[0].value_counts())


# ### **Splitting the data into Training and Test Data**

# In[56]:


X_train,X_test,y_train,y_test = train_test_split(x_resample,y_resample,test_size = 0.3, random_state=0)


# In[57]:


type(X_train),type(X_test),type(y_train),type(y_test)


# In[58]:


y_test = y_test.squeeze()
type(y_test)


# In[59]:


y_train = y_train.squeeze()
type(y_train)


# ### **Normalisation of the Data**

# In[60]:


se =StandardScaler()
normalised_X_train = se.fit_transform(X_train)
normalised_X_test = se.transform(X_test)
normalised_test_data = se.transform(test_set)


# In[61]:


Q = normalised_X_train
W = normalised_X_test


# ## **Model Building:**
# - The most important step is to build the model.
# - choosing a best model for out data, depends on various factors/metrics such as \
# accuracy, recall, precision, f1 score, etc.
# - In this case we are trying to implenmet 4 models and we are going to evaluate all the models, how they are performing and we will select the best model.
# - **DecisionTree, RandomForestClassifier, LogisticRegression, SVM.**

# In[62]:


# Function to calculate the accuracy of the model.

def print_score(clf, X_train, y_train, X_test, y_test, train=True):
    if train:
        pred = clf.predict(X_train)
        clf_report = pd.DataFrame(classification_report(y_train, pred, output_dict=True))
        print("Train Result:\n================================================")
        print(f"Accuracy Score: {accuracy_score(y_train, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_train, pred)}\n")
        
    elif train==False:
        pred = clf.predict(X_test)
        clf_report = pd.DataFrame(classification_report(y_test, pred, output_dict=True))
        print("Test Result:\n================================================")        
        print(f"Accuracy Score: {accuracy_score(y_test, pred) * 100:.2f}%")
        print("_______________________________________________")
        print(f"CLASSIFICATION REPORT:\n{clf_report}")
        print("_______________________________________________")
        print(f"Confusion Matrix: \n {confusion_matrix(y_test, pred)}\n")


# ### **Random Forest Classifier**

# In[63]:


rf_clf = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=None, min_samples_split=2, 
min_samples_leaf=1, min_weight_fraction_leaf=0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0, min_impurity_split=None, bootstrap=True, 
oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0, max_samples=None)

rf_clf.fit(Q, y_train)

print_score(rf_clf, Q, y_train, W, y_test, train=True)
print_score(rf_clf, Q, y_train, W, y_test, train=False)


# In[64]:


rf = RandomForestClassifier(n_estimators=100)
rf_model = rf.fit(Q,y_train)


# In[65]:


print("The Training accuracy using Random Forest Classifier is:", rf_model.score(Q,y_train))
print("The Testing Accuracy using Random Forest Classifier is:", rf_model.score(W,y_test))


# In[66]:


rf_pred = rf_model.predict(W)


# In[67]:


rf_pred


# In[68]:


print(confusion_matrix(y_test,rf_pred))


# In[69]:


plt.figure(figsize=(18,12))
plot_confusion_matrix(rf_model, normalised_X_test, y_test)
plt.xlabel("Predicted Label") 
plt.ylabel("Actual Label") 
plt.show()


# In[70]:


print(accuracy_score(y_test,rf_pred))


# In[71]:


print(recall_score(y_test,rf_pred))


# In[72]:


print(precision_score(y_test,rf_pred))


# In[73]:


from sklearn.metrics import f1_score


# In[74]:


print(f1_score(y_test,rf_pred))


# In[75]:


print(classification_report(y_test,rf_pred))


# In[ ]:





# ### **Logistic Regression**

# In[76]:


lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(Q, y_train)

print_score(lr_clf, Q, y_train, W, y_test, train=True)
print_score(lr_clf, Q, y_train, W, y_test, train=False)


# In[77]:


lr = LogisticRegression()
lr_model = lr.fit(Q, y_train)


# In[78]:


print("The Training accuracy using Logistic regression is:", lr_model.score(Q, y_train))
print("The Testing Accuracy using Logistic Regression is:", lr_model.score(W, y_test))


# In[79]:


lr_pred = lr_model.predict(W)


# In[80]:


print(recall_score(y_test,lr_pred))


# In[81]:


print(precision_score(y_test,lr_pred))


# In[82]:


print(f1_score(y_test,lr_pred))


# In[83]:


print(confusion_matrix(y_test, lr_pred))


# In[84]:


plt.figure(figsize=(18,12))
plot_confusion_matrix(lr_model, W, y_test)
plt.xlabel("Predicted Label") 
plt.ylabel("Actual Label") 
plt.show()


# In[85]:


print(classification_report(y_test,lr_pred))


# In[ ]:





# ### **Decision Tree Classifier**

# In[86]:


dt_clf = DecisionTreeClassifier()
dt_clf.fit(Q, y_train)

print_score(dt_clf, Q, y_train, W, y_test, train=True)
print_score(dt_clf, Q, y_train, W, y_test, train=False)


# In[87]:


dt = DecisionTreeClassifier()
dt_model = dt.fit(Q, y_train)


# In[88]:


print("The Training accuracy using Logistic regression is:", dt_model.score(Q, y_train))
print("The Testing Accuracy using Logistic Regression is:", dt_model.score(W, y_test))


# In[89]:


dt_pred = dt_model.predict(W)


# In[90]:


print(confusion_matrix(y_test, dt_pred))


# In[91]:


plt.figure(figsize=(18,12))
plot_confusion_matrix(dt_model, W, y_test)
plt.xlabel("Predicted Label") 
plt.ylabel("Actual Label") 
plt.show()


# In[92]:


print(recall_score(y_test,dt_pred))


# In[93]:


print(precision_score(y_test,dt_pred))


# In[94]:


print(f1_score(y_test,dt_pred))


# In[ ]:





# ### **Testing the Data**

# In[95]:


X_test.iloc[4]


# In[96]:


new = X_test.iloc[4]


# In[97]:


a = np.asarray(new)
a = a.reshape(1,-1)
p = rf_model.predict(a)


# In[98]:


if (p[0] == 1):
    print("Credit Card Should not be Approved.")
else:
    print("Great! You can trust the person")


# In[ ]:





# ## **Conclusion**
# - **Data Gathering:** Dataset used in this project is the Credit Card Approval Detection Dataset, which was downloaded from Kaggle.
# - **Data Pre-processing:** Data was pre-processing (Identifying Null Values, Outliers and Irrelevant data values) by removing the columns that are not required for the analysis.
# - **Exploratory Data Analysis** is performed so as to get some insights from the dataset, such as, this step will give us some interesting patterns which can help us get better results while modelling.
# - **Data Transformation:** Data transformation is performed so as to convert all our categorical string variables into numerical data, which is also the most important step because the model which we try to build cannot be trained by string values, since the machine cannot interpret the string data types as we convert it into numerical, if possible, depending on the requirement. Problem we even normalize the data, so as to bring all the columns as one scale.
# - **Model Building:** OK, here we build our model, using various algorithms chosen depending on the project we are working on, before building our model, we need to make sure that, the data is split into train and test data, and you need to make sure that the test data is kept securely without leaking the test data information to the model, if that happens it might lead to overfitting which can be serious problem in real world applications.
# - **Model Evaluation:** So, the model is built and in order to choose the best model, we evaluate our model, using various metrics, such as Accuracy, Recall, Precision, Classification Report, Confusion Matrix, etc.
# 

# In[ ]:




