#!/usr/bin/env python
# coding: utf-8

# In[56]:


import pandas as pd
import numpy as np


# In[57]:


df2=pd.read_csv('artists.csv')


# In[58]:


df2


# In[59]:


df2['id'].count()


# In[60]:


df=pd.read_csv('tracks.csv')


# 📁 tracks.csv
# 
# Detailed data about individual tracks:
# 
# id: Unique identifier for each track.
# name: The name/title of the track.
# popularity: Track popularity score on Spotify (0–100).
# duration_ms: Duration of the track in milliseconds.
# explicit: Boolean indicator of whether the track has explicit content.
# artists: List of artist names featured in the track.
# id_artists: Corresponding list of artist IDs for reference and joins.
# release_date: The track’s release date.
# danceability, energy: Audio features calculated by Spotify’s algorithm, representing the musical feel of the track.

# In[62]:


df


# 🔍 How You Can Use This Dataset
# 
# You can use this data to explore and answer questions such as:
# Artist Popularity Trends: Who are the most followed or popular artists? Which genres dominate Spotify?
# Track Insights: How do explicit tracks compare with clean tracks in terms of popularity or energy?
# Genre & Era Evolution: How have track characteristics like danceability or energy changed over time?
# Audio Feature Correlations: What is the relationship between track popularity and Spotify’s audio features like energy or danceability?
# Music Discovery Models: Use the artist and track data together for building music recommendation systems or clustering tracks by audio mood.
# 
# Whether you're a data analyst, music tech enthusiast, or someone experimenting with Spotify data, this dataset provides a great starting point for in-depth analysis.

# In[64]:


df.columns


# In[65]:


df.info()


# In[66]:


df.columns


# In[67]:


df.shape


# In[68]:


df.head()


# In[69]:


#Check for missing values

df.isnull().sum()
df=df.dropna()


# In[ ]:





# In[ ]:





# In[70]:


df.shape


# In[71]:


#Converting release date to datetime format

df['release_date']=pd.to_datetime(df['release_date'],errors='coerce')


# In[72]:


df.describe()


# In[73]:


df['duration_min']=df['duration_ms']/60000


# In[74]:


df.describe()


# In[ ]:





# ## EDA

# In[76]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df['popularity'], bins=50)
plt.title('Track Popularity Distribution')
plt.show()


# In[77]:


df['popularity'].hist(bins=50)


# In[78]:


features = ['danceability', 'energy', 'speechiness', 'acousticness', 
            'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']

corr = df[features].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# In[79]:


features=['popularity','duration_min','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo']


# In[80]:


corr=df[features].corr()
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title('Feature corelation matrix')
plt.figure(figsize=(1, 9)) 
plt.tight_layout()
plt.show()


# In[81]:


sns.boxplot(x='explicit', y='popularity', data=df)
plt.title('Popularity: Explicit vs. Non-explicit Tracks')
plt.show()


# In[82]:


df['year'] = df['release_date'].dt.year
yearly_avg = df.groupby('year')[['danceability', 'energy', 'popularity']].mean().reset_index()

plt.figure(figsize=(12,6))
for feature in ['popularity']:
    plt.plot(yearly_avg['year'], yearly_avg[feature], label=feature)

plt.title('Trend of Danceability, Energy, and Popularity Over Time')
plt.xlabel('Year')
plt.ylabel('Average Value')
plt.legend()
plt.show()


# In[83]:


df['year'] = df['release_date'].dt.year
yearly_avg = df.groupby('year')[['danceability', 'energy', 'popularity']].mean().reset_index()

plt.figure(figsize=(12,6))
for feature in ['danceability']:
    plt.plot(yearly_avg['year'], yearly_avg[feature], label=feature)

plt.title('Trend of Danceability, Energy, and Popularity Over Time')
plt.xlabel('Year')
plt.ylabel('Average Value')
plt.legend()
plt.show()


# In[84]:


df.groupby('year')['id'].count().reset_index().plot()


# In[85]:


df['year'] = df['release_date'].dt.year
yearly_avg = df.groupby('year')[['danceability', 'energy', 'popularity']].mean().reset_index()

plt.figure(figsize=(12,6))
for feature in ['energy']:
    plt.plot(yearly_avg['year'], yearly_avg[feature], label=feature)

plt.title('Trend of Danceability, Energy, and Popularity Over Time')
plt.xlabel('Year')
plt.ylabel('Average Value')
plt.legend()
plt.show()


# In[86]:


df


# In[87]:


df


# In[88]:


df2.columns


# In[89]:


df2.rename({'id':'id'},inplace=True)


# In[90]:


df.head()


# In[91]:


#df = df.merge(df2[['id', 'followers']], on='id', how='left')


# In[ ]:





# In[ ]:





# In[ ]:





# In[92]:


df2[df2['popularity']>0]


# In[93]:


df.columns


# In[94]:


df.head()


# In[95]:


'35iwgR4jXetI318WEWsa1Q' in df['id']


# In[96]:


'Right the Stars' in df['name'].values


# In[97]:


df.columns


# In[98]:


df['id_artists']


# In[99]:


import ast

df['id_artists'] = df['id_artists'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['id_artists'] = df['id_artists'].apply(lambda x: x[0] if isinstance(x, list) else x)


# In[100]:


df.head()


# In[101]:


df2.head()


# In[102]:


#df2.rename(columns={'id': 'id_artists'}, inplace=True)
df2.rename(columns={'id':'id_artists'},inplace=True)


# In[103]:


df2.head()


# In[104]:


import ast

df['artists'] = df['artists'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['main_artist'] = df['artists'].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)


# In[105]:


df.head()


# In[106]:


df_final=df.merge(df2[['id_artists','followers']],on='id_artists',how='left')


# In[107]:


df[df['main_artist']=='Uli']


# In[108]:


df_final[df_final['main_artist']=='Uli']


# In[ ]:





# In[ ]:





# In[112]:


features = ['danceability', 'energy', 'speechiness', 'acousticness', 'instrumentalness',
            'liveness', 'valence', 'loudness', 'tempo']


# In[114]:


# 11. Clustering songs by audio features (KMeans example)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = df_final[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

df_final.loc[X.index, 'cluster'] = clusters

plt.figure(figsize=(10,6))
sns.scatterplot(x='danceability', y='energy', hue='cluster', data=df_final, palette='Set2')
plt.title('KMeans Clustering of Songs by Danceability and Energy')
plt.show()


# In[117]:


df_final.columns


# In[121]:





# In[125]:


# Check for null values
df_final.isnull().sum()


# In[135]:


df_clean = df_final.dropna(subset=['danceability', 'energy', 'loudness', 'instrumentalness', 'name', 'main_artist'])


# In[137]:


df_clean


# In[139]:


# Use only 4 key features for recommendation
feature_columns = ['danceability', 'energy', 'loudness', 'instrumentalness']


# In[150]:


X


# In[141]:


X = df_clean[feature_columns]


# In[148]:


from sklearn.neighbors import NearestNeighbors
# Build K-Nearest Neighbors model
model = NearestNeighbors(n_neighbors=5, algorithm='auto')
model.fit(X)

# Example Input Song (can be any random or specific song from dataset)
sample_song = X.iloc[10].values.reshape(1, -1)

# Find 5 nearest songs based on mood parameters
distances, indices = model.kneighbors(sample_song)

# Recommended Songs
recommended_songs = df_clean.iloc[indices[0]][['name', 'main_artist']]
print(recommended_songs)


# In[152]:





# In[162]:


#!pip install streamlit



# In[160]:


#df_final


# In[164]:


import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Load dataset
df_clean = df.dropna(subset=['danceability', 'energy', 'valence', 'tempo', 'name', 'main_artist'])

# Prepare KNN model
feature_columns = ['danceability', 'energy', 'loudness', 'instrumentalness']
X = df_clean[feature_columns]
model = NearestNeighbors(n_neighbors=5)
model.fit(X)

# Streamlit UI
st.title("🎵 Spotify Mood-Based Song Recommender")

st.write("Select your preferred mood parameters:")

# User inputs
danceability = st.slider('Danceability', 0.0, 1.0, 0.5)
energy = st.slider('Energy', 0.0, 1.0, 0.5)
valence = st.slider('Valence (Positivity)', 0.0, 1.0, 0.5)
tempo = st.slider('Tempo (BPM)', 50.0, 200.0, 120.0)

# Button to trigger recommendation
if st.button('Recommend Songs'):
    input_features = [[danceability, energy, valence, tempo]]
    distances, indices = model.kneighbors(input_features)
    
    recommended_songs = df_clean.iloc[indices[0]][['name', 'main_artist']]
    st.subheader("🎧 Recommended Songs:")
    st.dataframe(recommended_songs)


# In[170]:





# In[ ]:




