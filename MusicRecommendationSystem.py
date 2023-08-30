#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('dataset.csv')
df.drop(columns='Unnamed: 0', inplace=True)
year = pd.read_csv('data_by_year.csv')
df_year = pd.read_csv('data.csv')
df_year = df_year[['id', 'year']]
df_year['track_id'] = df_year['id']
df_year.drop(columns='id', inplace=True)
genres = pd.read_csv('data_by_genres.csv')
artists = pd.read_csv('data_by_artist.csv')

df = pd.merge(df,df_year, on='track_id')
#display(df.info(),df.head())


# In[3]:


df[df.duplicated('track_id')==True]


# In[4]:


df[df['track_id']=='3ILmwMefYZoQh5Cf5jeuUQ']


# In[5]:


xtab_song = pd.crosstab(df['track_id'], df['track_genre'])
xtab_song = xtab_song*2
#display(xtab_song.head(),len(xtab_song))


# In[6]:


dfDistinct = df.drop_duplicates('track_id')
dfDistinct = dfDistinct.sort_values('track_id')
dfDistinct = dfDistinct.reset_index(drop=True)

xtab_song.reset_index(inplace=True)
data_encoded = pd.concat([dfDistinct, xtab_song], axis=1)
#display(data_encoded.head(), len(data_encoded))


# In[7]:


numerical_features = ['explicit', 'danceability', 'energy', 'loudness', 
                     'speechiness', 'acousticness', 'instrumentalness', 
                     'liveness', 'valence', 'year']
scaler = MinMaxScaler()
data_encoded[numerical_features] = scaler.fit_transform(data_encoded[numerical_features])


# In[8]:


calculated_features = numerical_features + list(xtab_song.drop(columns='track_id').columns)

cosine_sim = cosine_similarity(data_encoded[calculated_features], data_encoded[calculated_features])


# In[9]:


sound_features = ['acousticness', 'danceability', 'energy', 
                 'instrumentalness', 'liveness', 'valence']
fig = px.line(year, x='year', y=sound_features)
fig.show()


# In[10]:


top_10_genres = genres.nlargest(10, 'popularity')
fig = px.bar(top_10_genres, x='genres', y=['valence', 'energy', 
                                           'acousticness',  
                                           'danceability'], barmode='group')
fig.show()


# In[11]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(
n_clusters=10))])
X = genres.select_dtypes(np.number)
cluster_pipeline.fit(X)
genres['cluster'] = cluster_pipeline.predict(X)


# In[12]:


from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', 
                                                        TSNE(n_components=2, verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genres['genres']
projection['cluster'] = genres['cluster']

fig = px.scatter(
projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()


# In[13]:


song_cluster_pipeline = Pipeline([('scaler', StandardScaler()), (
'kmeans', KMeans(n_clusters=12, verbose=False))], verbose=False)

X = df.select_dtypes(np.number)
cols = list(X.columns)
song_cluster_pipeline.fit(X)
song_cluster_labels = song_cluster_pipeline.predict(X)
df['cluster_label'] = song_cluster_labels


# In[14]:


from sklearn.decomposition import PCA

pca_pipeline = Pipeline([('scaler', StandardScaler()), ('PCA', 
                                                        PCA(n_components=2))])
song_embedding = pca_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=song_embedding)
projection['title'] = df['track_name']
projection['cluster'] = df['cluster_label']

fig = px.scatter(projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'title'])
fig.show()


# In[15]:


def get_recommendations(title, N=5):
    indices = pd.Series(data_encoded.index, index=data_encoded['track_name']).drop_duplicates()
    
    try:
        idx = indices[title]
        try:
            len(idx)
            temp = 2
        except:
            temp = 1
    except KeyError:
        return "Unfortunately that song cannot be found. Please try a different song."
    
    if temp == 2:
        idx = indices[title][0]
    else:
        idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:N+1]
    song_indices = [i[0] for i in sim_scores]
    recommended_songs = data_encoded[['track_name', 'artists', 'album_name']].iloc[song_indices]
    
    sim_scores_list = [i[1] for i in sim_scores]
    recommended_list = recommended_songs.to_dict(orient='records')
    for i, song in enumerate(recommended_list):
        song['similarity_score'] = sim_scores_list[i]
        
    return recommended_list


# In[16]:


# This section of code was for testing purposes
recommended_songs = get_recommendations("Chop Suey!", N=5)
if isinstance(recommended_songs, str):
    print(recommended_songs)
else:
    print("Recommended Songs:")
    for song in recommended_songs:
        print(f"Title: {song['track_name']}")
        print(f"Artist: {song['artists']}")
        print(f"Albume: {song['album_name']}")
        print(f"Similarity Score: {song['similarity_score']:.5f}")
        print()


# In[17]:


get_recommendations('Toxicity', 5)


# In[18]:


get_recommendations('Chop Suey!', 10)


# In[19]:


get_recommendations('Death of a Bachelor', 5)


# In[20]:


get_recommendations('Psychosocial', 5)


# In[21]:


get_recommendations('Almost Easy', 5)


# In[ ]:




