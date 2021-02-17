#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
from IPython.display import Image


#Import data, drop rows with missing values

path = '/Users/melissanebel/Documents/Data Visualization Final Project/archive-3/IMDb movies.csv'
df = pd.read_csv(path,low_memory=False)
df = df.dropna()


# In[2]:


#Check data
df.head()


# In[3]:


#Drop columns we're not going to be using
df = df.drop(columns=['imdb_title_id','title','usa_gross_income', 'original_title','date_published','director','writer','production_company','actors','description','reviews_from_users','reviews_from_critics'])
df.head()


# In[4]:


#Describe data and check data types
df.describe()
df.dtypes


# In[5]:


#Drop non-dollar currency values for comparison

df = df.drop(df[~df.budget.str.contains("\$")].index)
df = df.drop(df[~df.worlwide_gross_income.str.contains("\$")].index)


# In[6]:


#Clean currency data
df['budget'] = df['budget'].replace({'\$': '', ',': ''}, regex=True).astype(int)
df['worlwide_gross_income'] = df['worlwide_gross_income'].replace({'\$': '', ',': ''}, regex=True).astype(int)


# In[7]:


#Set appropriate data types
df.year = df.year.astype(int)
df.metascore = df.metascore.astype(int)


# In[8]:


#Recheck data
df.describe()


# In[9]:


#Check for duplicates
Duplicate_Rows= df[df.duplicated()]
print(Duplicate_Rows)


# In[10]:


#Drop additional countries, genres and languages for simplicity

df['genre'] = df.genre.apply(lambda x: x.split()[0])
df['country'] = df.country.apply(lambda x: x.split()[0])
df['language'] = df.language.apply(lambda x: x.split()[0])

df['genre'] = df['genre'].str.replace(r'\,', '')
df['country'] = df['country'].str.replace(r'\,', '')
df['language'] = df['language'].str.replace(r'\,', '')

df.head()


# In[11]:


#Check correlation between numerical characteristics

numeric_columns = df.columns[df.dtypes != 'object']
numeric_df = pd.DataFrame(data=df, columns=numeric_columns, index=df.index)
corr = np.abs(numeric_df.corr())
fig, ax = plt.subplots(figsize=(8, 8))
cmap = sns.color_palette("mako")
sns.heatmap(corr, cmap=cmap, square=True)
plt.title('Correlation between numerical characteristics')
plt.show()


# In[12]:



#Set up variables and create pipeline for Decision Tree Regressor

X = df.drop('metascore', axis='columns')
y = df.metascore


column_trans = make_column_transformer(
    (OneHotEncoder(handle_unknown = "ignore"),['genre', 'country','language']),
    remainder='passthrough')

clf = Pipeline(steps=[('column_trans',column_trans),
                     ('classifier', DecisionTreeRegressor())])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[13]:


#Finding best parameters for decision tree

from sklearn.model_selection import GridSearchCV
param_grid = {"classifier__criterion": ["mse", "mae"],
              "classifier__min_samples_split": [10, 20, 40],
              "classifier__max_depth": [2, 6, 8],
              "classifier__min_samples_leaf": [20, 40, 100],
              "classifier__max_leaf_nodes": [5, 20, 100],
              }

grid_cv_dtm = GridSearchCV(clf, param_grid, cv=5)

grid_cv_dtm.fit(X_train,y_train)

print("R-Squared::{}".format(grid_cv_dtm.best_score_))
print("Best Parameters::\n{}".format(grid_cv_dtm.best_params_))


# In[14]:


df = pd.DataFrame(data=grid_cv_dtm.cv_results_)
df.head()


# In[15]:


fig,ax = plt.subplots()
sns.pointplot(data=df[['mean_test_score',
                           'param_classifier__max_leaf_nodes',
                           'param_classifier__max_depth']],
             y='mean_test_score',x='param_classifier__max_depth',
             hue='param_classifier__max_leaf_nodes',ax=ax)
ax.set(title="Effect of Depth and Leaf Nodes on Model Performance")


# In[16]:


grid_cv_dtm.best_estimator_


# In[19]:


#Visualize decision tree from best estimators

dot_data = StringIO()

export_graphviz(grid_cv_dtm.best_estimator_['classifier'],
                max_depth=3,
                out_file=dot_data,
                feature_names=None,
                class_names=None,
                filled=True,
                rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())


# In[ ]:




