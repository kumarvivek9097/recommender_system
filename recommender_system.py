import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

#Plot Description Based Recommender

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

#Credits, Genres and Keywords Based Recommender

credits= pd.read_csv('/home/vivek/Documents/Data_Science/Data_Sets/Imdb/credits.csv')
keywords=pd.read_csv('/home/vivek/Documents/Data_Science/Data_Sets/Imdb/keywords.csv')

metadata['id']=metadata['id'].astype('int')
keywords['id']=keywords['id'].astype('int')
credits['id']=credits['id'].astype('int')
metadata=metadata.merge(keywords, on='id')
metadata=metadata.merge(credits,on='id')

features = ['cast', 'crew', 'keywords', 'genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(literal_eval)

def get_director(x):
    for i in x:
        if i['job']=="Director":
            return i['name']
    return np.nan

def get_list(x):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        if len(names) > 3:
            names = names[:3]
        return names
    return []

metadata['director']=metadata['crew'].apply(get_director)
features=['cast','keywords','genres']
for feature in features:
    metadata[feature] = metadata[feature].apply(get_list)

def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

features = ['cast', 'keywords', 'director', 'genres']

for feature in features:
    metadata[feature] = metadata[feature].apply(clean_data)

def create_soup(x):
    return ' '.join(x['keywords']) + ' ' + ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

metadata['soup'] = metadata.apply(create_soup, axis=1)
count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(metadata['soup'])

cosine_sim2 = cosine_similarity(count_matrix, count_matrix)

metadata = metadata.reset_index()
indices = pd.Series(metadata.index, index=metadata['title'])
get_recommendations('The Dark Knight Rises', cosine_sim2)
