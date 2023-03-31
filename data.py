# Harel Meir 205588940
import sys
import matplotlib.pyplot as plt
import seaborn as sns

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    ratings = data[0]
    """   1.1  """
    unique_users_list = ratings['userId'].unique()
    num_unique_users = len(unique_users_list)   # 21244
    print("Number of unique users: " + str(num_unique_users))


    """  1.2   """
    unique_movies_list = ratings['movieId'].unique()
    num_unique_movies = len(unique_movies_list)   # 1342
    print("Number of unique movies: " + str(num_unique_movies))

    """  1.3   """
    num_ratings = len(ratings)   # 3720552
    print("Number of ratings: " + str(num_ratings))

    """   2   """
    movies_occ = ratings['movieId'].value_counts()
    m_max_ratings = movies_occ.max()    # 9416
    m_min_rating = movies_occ.min()     # 462
    print("Smallest number of unique movies ratings: " + str(m_min_rating))
    print("Biggest number of unique movies ratings   " + str(m_max_ratings))

    """  3   """
    users_occ = ratings['userId'].value_counts()
    u_max_ratings = users_occ.max()   # 432
    u_min_ratings = users_occ.min()   # 46
    print("Smallest number of unique user ratings: " + str(u_min_ratings))
    print("Biggest number of unique user ratings:  " + str(u_max_ratings))


def plot_data(data, plot = True):
    # sorting by rating.
    if plot:
        ratings = data[0]
        ratings = ratings.sort_values(by=['rating'])
        # ploting the data.
        ratings['rating'].value_counts(sort=False).plot.bar()
        plt.xlabel("ratings")
        plt.ylabel("Amount")
        plt.show()

