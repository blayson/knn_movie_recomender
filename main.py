# %%

import numpy as np
import pandas as pd
from scipy import spatial

MEAN = 23
SIZE = 22
TITLE = 1


def compute_distance(a: list, b: list):
    genres_a = a[0]
    genres_b = b[0]
    genre_distance = spatial.distance.euclidean(genres_a, genres_b)
    popularity_a = a[1]
    popularity_b = b[1]
    popularity_distance = abs(popularity_a - popularity_b)
    return genre_distance + popularity_distance


def get_neighbors(movie_id: int, k: int, data: pd.DataFrame):
    distances = []
    query = [list(data.loc[data['movieId'] == movie_id].iloc[0][2:22]),
             data.loc[data['movieId'] == movie_id].iloc[0][SIZE]]
    for index, row in data.iterrows():
        if row['movieId'] != movie_id:
            dist = compute_distance(query, [row[2:22], row[SIZE]])
            distances.append((row['movieId'], dist))
    distances.sort(key=lambda item: item[1])
    neighbors = distances[:k]
    return neighbors


def main(movie_id: int, k: int):
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

    avg_rating = 0
    print(movies_and_ratings.loc[movies_and_ratings['movieId'] == movie_id])
    neighbors = get_neighbors(movie_id=movie_id, k=k, data=movies_and_ratings)
    for neighbor in neighbors:
        avg_rating += movies_and_ratings.loc[movies_and_ratings['movieId'] == neighbor[0]].iloc[0][MEAN]
        title = movies_and_ratings.loc[movies_and_ratings['movieId'] == neighbor[0]].iloc[0][TITLE]
        mean = movies_and_ratings.loc[movies_and_ratings['movieId'] == neighbor[0]].iloc[0][MEAN]
        print(f"{title} {mean}")

    avg_rating /= k
    print(f"Average rating: {avg_rating}")
    print(neighbors)
    return neighbors


if __name__ == '__main__':
    main(movie_id=1, k=10)  # Toy Story (1995)


