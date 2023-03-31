# Harel Meir 205588940

from math import sqrt
# Import Pandas
import pandas as pd
import numpy as np


# we get top 10 reccomnedation for the user.
def precision_10(test_set, cf, is_user_based = True):
    test_data_set = pd.pivot_table(test_set, index=['userId'], values='rating', columns=['movieId'], aggfunc=np.sum)
    val = 0
    # iterating over each unique user.
    for unique_user in test_data_set.index:
        # getting the movies
        user_row = test_data_set.loc[unique_user]
        # getting top_k prediction for the user.
        top_k_movies_ids = cf.top_k_ids(unique_user, 10,is_user_based)
        # getting the test movies ids that are larger then 4 rating.
        test_movies_ids = user_row[user_row >= 4.0].index

        val += int(len(set(top_k_movies_ids) & set(test_movies_ids))) / 10
    val = val / len(test_data_set.index)
    print("Precision_k: " + str(val))


def ARHA(test_set, cf, is_user_based = True):
    test_data_set = pd.pivot_table(test_set, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.sum)
    sum = 0
    for user_id in test_data_set.index:
        user_row = test_data_set.loc[user_id]
        test_ids = user_row[user_row >= 4.0].index
        top_k = cf.top_k_ids(user_id, 10, is_user_based)
        for pos, movie_id in enumerate(top_k):
            if movie_id in test_ids:
                pos += 1
                sum += 1 / pos

    val = sum / len(test_data_set)
    print("ARHR: " + str(val))


def RSME(test_set, cf, is_user_based = True):
    test_data_set = pd.pivot_table(test_set, values='rating', index=['userId'], columns=['movieId'], aggfunc=np.sum)
    numerator = 0
    denominator = 0
    for unique_user in test_data_set.index:
        user_row = test_data_set.loc[unique_user]
        # calculating the subtraction
        subs = cf.ratings_prediction(unique_user, is_user_based) - user_row.values
        # powering ** 2
        after_power = np.power(subs, 2)

        sum_nans = np.nansum(after_power)
        numerator += sum_nans
        # count number of movies that were considered
        sum_not_nans = np.sum(~np.isnan(after_power))
        denominator += sum_not_nans

    val = sqrt(numerator/denominator)
    print("RMSE: " + str(val))


