# %%
import numpy as np
import pandas as pd
from scipy import spatial

MEAN = 23
SIZE = 22
TITLE = 1


def compute_euclidean_distance(first: list, second: list) -> float:
    """Return euclidean distance between two arrays + popularity distance"""
    genres_a = first[0]
    genres_b = second[0]
    genre_distance = spatial.distance.euclidean(genres_a, genres_b)
    popularity_a = first[1]
    popularity_b = second[1]
    popularity_distance = abs(popularity_a - popularity_b)
    return genre_distance + popularity_distance


def get_neighbors(movie_id: int, k: int, data: pd.DataFrame) -> list:
    """Return neighbors based on knn algorithm"""
    distances = []
    query = [list(data.loc[data['movieId'] == movie_id].iloc[0][2:22]),
             data.loc[data['movieId'] == movie_id].iloc[0][SIZE]]
    for index, row in data.iterrows():
        if row['movieId'] != movie_id:
            dist = compute_euclidean_distance(query, [row[2:22], row[SIZE]])
            distances.append((row['movieId'], dist))
    distances.sort(key=lambda item: item[1])
    neighbors = distances[:k]
    return neighbors


def prepare_data() -> pd.DataFrame:
    ratings = pd.read_csv('ml_data_small/ratings.csv', usecols=range(3))
    movie_properties = ratings.groupby('movieId').agg({'rating': [np.size, np.mean]})
    movie_num_ratings = pd.DataFrame(movie_properties['rating']['size'])
    movie_normalized_num_ratings = movie_num_ratings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

    movies = pd.read_csv('ml_data_small/movies.csv')
    movies = pd.concat([movies, movies['genres'].str.get_dummies(sep='|')], axis=1)
    movies.drop('genres', axis=1, inplace=True)
    movies_and_ratings = pd.merge(movies,
                                  movie_normalized_num_ratings,
                                  on='movieId',
                                  how='left')
    movies_and_ratings = pd.merge(movies_and_ratings,
                                  pd.DataFrame(movie_properties['rating']['mean']),
                                  on='movieId',
                                  how='left')
    return movies_and_ratings


def recommend_movies(movie_id: int, k: int, data: pd.DataFrame) -> list:
    kneighbors = get_neighbors(movie_id=movie_id, k=k, data=data)
    avg_rating = 0
    print('============== Recommended movies ==============')
    for neighbor in kneighbors:
        movie = data['movieId'] == neighbor[0]
        mean = data.loc[movie].iloc[0][MEAN]
        title = data.loc[movie].iloc[0][TITLE]
        avg_rating += mean
        print(f"Title: {title}, rating: {mean}")

    print('===============================================')
    avg_rating /= 10
    print(f"Average rating: {avg_rating}")
    return kneighbors


if __name__ == '__main__':
    m_id = 1

    movies_data = prepare_data()
    print(movies_data.loc[movies_data['movieId'] == m_id])

    recommended_movies = recommend_movies(movie_id=m_id, k=10, data=movies_data)  # Toy Story (1995)
