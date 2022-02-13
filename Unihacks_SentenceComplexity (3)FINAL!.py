#!/usr/bin/env python
# coding: utf-8

# ## End-to-End Pipeline for Machine Learning: Classification

# In[1]:


# foundational modules
import numpy as np
import pandas as pd

# visualization
import matplotlib.pyplot as plt 
import matplotlib.figure as fig 
import seaborn as sns
sns.set(style="darkgrid")

import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.corpus import stopwords
nltk.download("stopwords")

import string

# data preparation for model learning
from sklearn.model_selection import train_test_split

# model building
from sklearn.linear_model import LogisticRegression

# model building
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

### classification metrics
from sklearn.metrics import classification_report

def _syllables(word):
    syllable_count = 0
    vowels = 'aeiouy'
    if word[0] in vowels:
        syllable_count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            syllable_count += 1
    if word.endswith('e'):
        syllable_count -= 1
    if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        syllable_count += 1
    if syllable_count == 0:
        syllable_count += 1
    return syllable_count


# ### Step-1: Load and Examine the Data Set

# In[2]:


df = pd.read_csv("data_to_train.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.describe()


# ### Step-2: Split Data into Train, Validation, and Test Data Sets

# In[5]:


y = np.array ( df.level)
X = np.array ( df.drop ( columns = ['level'] ) )

def stratified_split(X, y, 
                     test_size=0.2, 
                     validate_size=0.2, 
                     random_state=0):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state = random_state)

    # need to calculate new split size. 
    # let's assume we had 100 samples and we don't do this
    # then the split will be 20 + (20% of 80) + (80% of 80). 
    # But we want 20 + 20 + 60
    new_validate_size = validate_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, stratify=y_train, 
        test_size=new_validate_size, 
        random_state = random_state)

    return X_train, X_test, X_val, y_train, y_test, y_val


# In[6]:


# Split data into training, validation, and testing data sets
X_train, X_test, X_val, y_train, y_test, y_val = stratified_split (
    X, y, random_state = 66)


# In[7]:


# Examine the split proportions 

print ("Training (X_train and y_train): \t", X_train.shape, " \t", y_train.shape)
print ("Validation (X_val and y_val): \t\t", X_val.shape, " \t", y_val.shape)
print ("Testing (X_test and y_test): \t\t", X_test.shape, "  \t", y_test.shape)


# ### Step-3: Define the Parameters for Single Model Classifiers and Fit the Model

# In[8]:


# Set-up and Build SVM classifier model
svc = SVC(gamma='auto')
svc.fit(X_train, y_train)


# In[9]:


# Set-up and Build Decision Tree classifier model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[10]:


# Set-up and Build Logistic Regression model
lr = LogisticRegression(solver='newton-cg')
lr.fit(X_train, y_train)


# In[11]:


# Set-up and Build Gaussian NB model
nb = GaussianNB()
nb.fit(X_train, y_train)


# In[12]:


# Set-up and Build k-Nearest Neighbor model
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)


# ### Step-4: Report Performance Metrics for the Training, Validation and Test Data

# In[13]:


# SVM model performance
n_sv = np.sum(svc.n_support_)

print("SVM Performance:")
print ("\tTRAIN Accuracy: {:.2f}".format(svc.score(X_train, y_train)))
print ("\tVALIDATION Accuracy: {:.2f}".format(svc.score(X_val, y_val)))
print ("\tSupport Vectors: {:.0f}% out of all the training points".format(100 * n_sv / X_train.shape[0]))
print ("\tTEST Accuracy: {:.2f}".format(svc.score(X_test, y_test)))


# In[14]:


# Logistic Regression model performance
print("Logistic Regression Performance:")
print ("\tTRAIN Accuracy: {:.2f}".format(lr.score(X_train, y_train)))
print ("\tVALIDATION Accuracy: {:.2f}".format(lr.score(X_val, y_val)))
print ("\tTEST Accuracy: {:.2f}".format(lr.score(X_test, y_test)))


# In[15]:


# Naive Bayes Classifier model performance
print("Naive Bayes Performance:")
print ("\tTRAIN Accuracy: {:.2f}".format(nb.score(X_train, y_train)))
print ("\tVALIDATION Accuracy: {:.2f}".format(nb.score(X_val, y_val)))
print ("\tTEST Accuracy: {:.2f}".format(nb.score(X_test, y_test)))


# In[16]:


# k-Nearest Neighbor Classifier model performance
print("KNN Performance:")
print ("\tTRAIN Accuracy: {:.2f}".format(knn.score(X_train, y_train)))
print ("\tVALIDATION Accuracy: {:.2f}".format(knn.score(X_val, y_val)))
print ("\tTEST Accuracy: {:.2f}".format(knn.score(X_test, y_test)))


# In[17]:


# Decision Tree model performance
print("Decision Tree Performance:")
print ("\tTRAIN Accuracy: {:.2f}".format(dt.score(X_train, y_train)))
print ("\tVALIDATION Accuracy: {:.2f}".format(dt.score(X_val, y_val)))
print ("\tTEST Accuracy: {:.2f}".format(dt.score(X_test, y_test)))


# ### Step-5: Play with Different Parameters to Improve each Model

# In[18]:


# Linear SVM model
svc = SVC(kernel="linear")
svc.fit(X_train, y_train)


# In[19]:


print("Linear SVM Performance:")
print ("\tTRAIN Accuracy: {:.2f}".format(svc.score(X_train, y_train)))
print ("\tVALIDATION Accuracy: {:.2f}".format(svc.score(X_val, y_val)))
print ("\tTEST Accuracy: {:.2f}".format(svc.score(X_test, y_test)))


# In[20]:


# k-NN Classifier with k=3 neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("KNN Performance:")
print ("\tTRAIN Accuracy: {:.2f}".format(knn.score(X_train, y_train)))
print ("\tVALIDATION Accuracy: {:.2f}".format(knn.score(X_val, y_val)))
print ("\tTEST Accuracy: {:.2f}".format(knn.score(X_test, y_test)))


# In[21]:


# Examine Feature Importance
feature_names = df.drop ( columns = ['level']).columns
fdf = pd.DataFrame(data = list(zip(feature_names, dt.feature_importances_)), 
                   columns = ["Feature Names", "Feature Importances"])
fdf.head(3)


# In[22]:


sns.barplot(y = "Feature Names", x="Feature Importances", data = fdf,
           color="salmon", saturation=1.0)
plt.show()


# ### Select the Best Model

# In[23]:


SVM_predictionsValidate = svc.predict(X_val)
print (classification_report(y_val, SVM_predictionsValidate))


# In[24]:


LR_predictionsValidate = lr.predict(X_val)
print (classification_report(y_val, LR_predictionsValidate))


# In[25]:


NB_predictionsValidate = nb.predict(X_val)
print (classification_report(y_val, NB_predictionsValidate))


# In[26]:


KNN_predictionsValidate = knn.predict(X_val)
print (classification_report(y_val, KNN_predictionsValidate))


# In[27]:


DT_predictionsValidate = dt.predict(X_val)
print (classification_report(y_val, DT_predictionsValidate))


# ### Step-6: Predict the Level of Complexity of a Given Sentence

# In[28]:


user = """I love running."""
user_string = user.translate(str.maketrans('', '', string.punctuation))
sent_tokenize(user_string)
user_words = word_tokenize(user_string)
user_list = [word for word in user_words if not word in stopwords.words()]
user_word_count = len(user_list)
user_char_count = 0
for word in user_list:
    user_char_count = user_char_count + len(word)
user_sly_count = 0
for word in user_list:
    user_sly_count = user_sly_count + _syllables(word)
user_data = [[user_word_count, user_char_count,user_sly_count]]
X = pd.DataFrame(user_data, columns = ['word_count', 'char_count', 'sly_count'])


# In[29]:


MyPrediction = svc.predict(X)
MyPrediction[0]


# In[ ]:





# In[ ]:




