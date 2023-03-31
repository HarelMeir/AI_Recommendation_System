# Harel Meir 205588940
import heapq
import numpy as np
# Import Pandas
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances


def keep_top_k(arr, k):
    smallest = heapq.nlargest(k, arr)[-1]
    arr[arr < smallest] = 0  # replace anything lower than the cut off with 0
    return arr


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.users = []
        self.movies = []

    def create_fake_user(self, rating):
        "*** YOUR CODE HERE ***"
        d = {'userId': [283238, 283238, 283238, 283238, 283238],
             'movieId': [62, 141, 65, 69, 88],
             'rating': [5.0, 4.5, 5.0, 4.5, 5.0]}
        rating = rating.append(pd.DataFrame(data=d), ignore_index=True)
        return rating

    def create_user_based_matrix(self, data):
        # parsing the data
        ratings = data[0]
        self.movies = data[1]
        # for adding fake user
        ratings = self.create_fake_user(ratings)

        # create a pivot table.
        self.ratings_pd = pd.pivot_table(ratings, values='rating', index=['userId'], columns=['movieId'],
                                         aggfunc=np.sum)

        # calculating the mean of each user.
        mean_user_rating = self.ratings_pd.mean(axis=1).to_numpy().reshape(-1, 1)

        # calculating the rating diff matrix.
        ratings_diff = (self.ratings_pd - mean_user_rating)

        # setting nans to 0
        ratings_diff[np.isnan(ratings_diff)] = 0
        # ratings_diff.round(2)

        # cosine.
        user_similarity = 1 - pairwise_distances(ratings_diff, metric='cosine')
        # to avoid divide by 0
        np.seterr(divide='ignore', invalid='ignore')
        # return the user matrix.
        self.user_based_matrix = mean_user_rating + user_similarity.dot(ratings_diff) / np.array(
            [np.abs(user_similarity).sum(axis=1)]).T

    def create_item_based_matrix(self, data):
        # same as user matrix, for items.
        ratings = data[0]
        self.ratings_pd = pd.pivot_table(ratings, values='rating', index=['userId'], columns=['movieId'],
                                         aggfunc=np.sum)
        mean_user_rating = self.ratings_pd.mean(axis=1).to_numpy().reshape(-1, 1)
        ratings_diff = (self.ratings_pd - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)] = 0
        rating_item = ratings_diff
        item_similarity = 1 - pairwise_distances(rating_item.T, metric='cosine')
        np.seterr(divide='ignore', invalid='ignore')
        self.item_based_metrix = mean_user_rating + rating_item.dot(item_similarity) / np.array(
            [np.abs(item_similarity).sum(axis=1)])
        self.item_based_metrix = self.item_based_metrix.to_numpy()

    def predict_movies(self, user_id, k, is_user_based=True):
        # getting the top_k id's
        user_row = self.ratings_prediction(user_id, is_user_based)
        # convert to df so we can use nlargest and keep it's order
        temp_df = pd.DataFrame(user_row)
        # convert indicies to movie ids
        movies_indexes = self.ratings_pd.columns.values
        top_k = movies_indexes[temp_df.nlargest(k, 0).index]
        return np.flip(self.movies.loc[self.movies['movieId'].isin(top_k)]['title'].values)

    def top_k_ids(self, user_id, k, is_user_based):
        user_row = self.ratings_prediction(user_id, is_user_based)
        # convert to df so we can use nlargest and keep it's order
        temp_df = pd.DataFrame(user_row)
        # convert indicies to movie ids
        movies_indexes = self.ratings_pd.columns.values
        top_k_ids = movies_indexes[temp_df.nlargest(k, 0).index]
        return top_k_ids

    def ratings_prediction(self, user_id, is_user_based):
        user_id = int(user_id)
        user_indexes = self.ratings_pd.index
        # get row of ratings for user_id
        user_row_ratings = self.ratings_pd.loc[user_id].to_numpy()
        # get user row from relevant base matrix
        user_list = user_indexes.to_list()
        if is_user_based:
            user_row = self.user_based_matrix[user_list.index(user_id)]
        else:
            user_row= self.item_based_metrix[user_list.index(user_id)]

        return (np.isnan(user_row_ratings)) * user_row
