

import numpy as np
import pandas as pd
import warnings


warnings.filterwarnings('ignore')


cl_nm=["user_id","item_id","rating","timestamp"]
df=pd.read_csv("u.data",sep='\t',names=cl_nm)


df.head()


df.shape

df["user_id"].nunique()


df["item_id"].nunique()


df1=pd.read_csv("u.item",sep='\|',header=None)


df1.head()

mv_tl=df1[[0,1]]


mv_tl.columns=['item_id','title']

mv_tl


ldf=pd.merge(df,mv_tl,on="item_id")

ldf



import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("white")


ldf.groupby("title").mean()

ldf.groupby("title").mean()['rating'].sort_values(ascending=False)


ldf.groupby("title").count()


ldf.groupby("title").count()['rating'].sort_values(ascending=False)


ratings=pd.DataFrame(ldf.groupby("title").mean()['rating'])


ratings


ratings["number of views"]=ldf.groupby("title").count()['rating']


ratings

ratings.sort_values(by='rating',ascending=False)


plt.figure(figsize=(10,6))
plt.hist(ratings["number of views"],bins=70)
plt.show()


plt.hist(ratings["rating"],bins=70)
plt.show()


sns.jointplot(x='rating',y='number of views',data=ratings,alpha=0.5)


ldf


nv=ldf.pivot_table(index="user_id",columns="title",values='rating')

nv.head()


ratings.sort_values('number of views',ascending =False).head()


star_war_us_id=nv['Star Wars (1977)']


sm=nv.corrwith(star_war_us_id)


corr_sw=pd.DataFrame(sm,columns=['Correlation'])


corr_sw.dropna(inplace=True)

corr_sw

corr_sw=corr_sw.sort_values('Correlation',ascending =False)


corr_sw=corr_sw.join(ratings['number of views'])


corr_sw[corr_sw['number of views']>100]



def pre_mv(mv_nm):
    mv_us_rt=nv[mv_nm]
    sm=nv.corrwith(mv_us_rt)
    
    corr_mv=pd.DataFrame(sm,columns=['Correlation'])
    corr_mv.dropna(inplace=True)
    
    corr_mv=corr_mv.join(ratings['number of views'])
    corr_mv=corr_mv.sort_values('Correlation',ascending =False)
    prediction=corr_mv[corr_mv['number of views']>100]
    
    return prediction


pr=pre_mv('Raiders of the Lost Ark (1981)')



pr




