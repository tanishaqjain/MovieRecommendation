#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import warnings


# In[3]:


warnings.filterwarnings('ignore')


# In[4]:


cl_nm=["user_id","item_id","rating","timestamp"]
df=pd.read_csv("u.data",sep='\t',names=cl_nm)


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df["user_id"].nunique()


# In[8]:


df["item_id"].nunique()


# In[9]:


df1=pd.read_csv("u.item",sep='\|',header=None)


# In[10]:


df1.head()


# In[11]:


mv_tl=df1[[0,1]]


# In[12]:


mv_tl.columns=['item_id','title']


# In[13]:


mv_tl


# In[14]:


ldf=pd.merge(df,mv_tl,on="item_id")


# In[15]:


ldf


# In[ ]:





# In[ ]:





# In[16]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


# In[17]:


ldf.groupby("title").mean()


# In[18]:


ldf.groupby("title").mean()['rating'].sort_values(ascending=False)


# In[19]:


ldf.groupby("title").count()


# In[20]:


ldf.groupby("title").count()['rating'].sort_values(ascending=False)


# In[21]:


ratings=pd.DataFrame(ldf.groupby("title").mean()['rating'])


# In[22]:


ratings


# In[23]:


ratings["number of views"]=ldf.groupby("title").count()['rating']


# In[24]:


ratings


# In[25]:


ratings.sort_values(by='rating',ascending=False)


# In[26]:


plt.figure(figsize=(10,6))
plt.hist(ratings["number of views"],bins=70)
plt.show()


# In[27]:


plt.hist(ratings["rating"],bins=70)
plt.show()


# In[28]:


sns.jointplot(x='rating',y='number of views',data=ratings,alpha=0.5)


# In[29]:


ldf


# In[30]:


nv=ldf.pivot_table(index="user_id",columns="title",values='rating')


# In[31]:


nv.head()


# In[33]:


ratings.sort_values('number of views',ascending =False).head()


# In[37]:


star_war_us_id=nv['Star Wars (1977)']


# In[39]:


sm=nv.corrwith(star_war_us_id)


# In[40]:


corr_sw=pd.DataFrame(sm,columns=['Correlation'])


# In[42]:


corr_sw.dropna(inplace=True)


# In[43]:


corr_sw


# In[47]:


corr_sw=corr_sw.sort_values('Correlation',ascending =False)


# In[49]:


corr_sw=corr_sw.join(ratings['number of views'])


# In[50]:


corr_sw[corr_sw['number of views']>100]


# ## Prediction Function

# In[55]:


def pre_mv(mv_nm):
    mv_us_rt=nv[mv_nm]
    sm=nv.corrwith(mv_us_rt)
    
    corr_mv=pd.DataFrame(sm,columns=['Correlation'])
    corr_mv.dropna(inplace=True)
    
    corr_mv=corr_mv.join(ratings['number of views'])
    corr_mv=corr_mv.sort_values('Correlation',ascending =False)
    prediction=corr_mv[corr_mv['number of views']>100]
    
    return prediction


# In[56]:


pr=pre_mv('Raiders of the Lost Ark (1981)')


# In[57]:


pr


# In[ ]:




