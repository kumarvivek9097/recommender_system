import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

metadata= pd.read_csv('/home/vivek/Documents/Data_Science/Data_Sets/movies_metadata.csv', low_memory=False)
c= metadata['vote_average'].mean()
m= metadata['vote_count'].quantile(0.90)
q_movies=metadata.copy().loc[metadata['vote_count']>=m]

def weighted_mean(x,m=m,c=c):
    v=x['vote_count']
    r=x['vote_average']
    return (v/(v+m)*r)+(m/(v+m)*c)

q_movies = q_movies.sort_values('score', ascending=False)
q_movies[['original_title','vote_count','vote_average','score']].head(10)

tfidf=TfidfVectorizer(stop_words='english')
metadata['overview']=metadata['overview'].fillna(' ')
tfidf_matrix=tfidf.fit_transform(metadata['overview'])

cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

indices=pd.Series(metadata.index,index=metadata['title']).drop_duplicates()

def get_recommendations(title,cosine_sim=cosine_sim):
        idx = indices[title]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]
        movie_indices = [i[0] for i in sim_scores]
        return metadata['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')

