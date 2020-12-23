# Importing Necessary Libraries Like lenskit, pandas

import lenskit.datasets as ds
from lenskit.algorithms import Recommender
from lenskit.algorithms.user_knn import UserUser
import pandas as pd
import csv

# Storing All The Datasets Files From Movielens dataset into our variable using lenskit
# and connecting the datasets

data_set = ds.MovieLens('Resource-File')
print("Successfully Connected Datasets... Welcome to our movie recommender system...")

# We will use ratings dataset file, from the table of that file, we will group
# all the ratings of a movie by movieID and then will find the average of it using mean()
# We will also check if a movie is rated by at least a certain of people
# in our case it's 50 users, as we cannot rely on 1/5 users for a reliable ratings
# then by sorting in descending based on ratings we will give a generic recommendation

rows_to_display = 10
minimum_user_to_include = 30

average_ratings = data_set.ratings.groupby(["item"]).mean()
ratings_count = data_set.ratings.groupby(["item"]).count()
average_ratings = average_ratings.loc[ratings_count["rating"] > minimum_user_to_include]  # filtering out rows
sorted_average_ratings = average_ratings.sort_values(by="rating", ascending=False)  # Sorting in descending by rating
joined_data = sorted_average_ratings.join(data_set.movies["genres"], on="item")
joined_data = joined_data.join(data_set.movies["title"], on="item")  # joining title column from movies file
joined_data = joined_data[joined_data.columns[3:]]
print("")
print("---------------------------------------------------------------------------------------------------------------")
print("Best 10 Recommended Movies Of All Time: ")
print("")
print(joined_data.head(rows_to_display).to_string())
print("")
print("---------------------------------------------------------------------------------------------------------------")


# Now we will take the input from user about his favourite movie genre and
# then we filter out the movies of that genre using contains and will provide
# a genre based movie recommendation

user_genre = input("Enter Your Favourite Movie Genre (Ex: Action/Romance/Adventure): ")  # Taking input from user

average_ratings = data_set.ratings.groupby(["item"]).mean()
ratings_count = data_set.ratings.groupby(["item"]).count()
average_ratings = average_ratings.loc[ratings_count["rating"] > minimum_user_to_include]
average_ratings = average_ratings.join(data_set.movies["genres"], on="item")
average_ratings = average_ratings.loc[average_ratings["genres"].str.contains(user_genre)]

sorted_average_ratings = average_ratings.sort_values(by="rating", ascending=False)
joined_data = sorted_average_ratings.join(data_set.movies["title"], on="item")
joined_data = joined_data[joined_data.columns[3:]]
print("")
print("---------------------------------------------------------------------------------------------------------------")
print("Best 10 Recommended " + user_genre + " Movies Of All Time: ")
print("")
print(joined_data.head(rows_to_display).to_string())
print("")
print("---------------------------------------------------------------------------------------------------------------")

# Creating two dictionaries to store my and my friend watched rated movie data

swapnil_ratings_dict = {}
my_friend_ratings_dict = {}

# with the help of csv reader, reading my movie ratings csv file and storing data to my dict

with open("Resource-File/swapnil-ratings.csv", newline='') as csvFile:
    ratings_reader = csv.DictReader(csvFile)
    for row in ratings_reader:
        row_ratings = row["ratings"]
        if(row_ratings != "") and (float(row_ratings) > 0) and (float(row_ratings) < 6):
            swapnil_ratings_dict.update({int(row["item"]): float(row_ratings)})

# with the help of csv reader, reading my friend's movie ratings csv file and storing data to his dict

with open("Resource-File/my-friend-ratings.csv", newline='') as csvFile:
    ratings_reader = csv.DictReader(csvFile)
    for row in ratings_reader:
        row_ratings = row["ratings"]
        if(row_ratings != "") and (float(row_ratings) > 0) and (float(row_ratings) < 6):
            my_friend_ratings_dict.update({int(row["item"]): float(row_ratings)})

# By using the user-user collaborator algorithm, we are finding users who like same kind of movie as me
# and then recommending their favourite movie i didn't watched, as we a same kind of taste
# we select user withing a defined range/ neighborhood and give a personal movie recommendation for myself

num_represent = 15

user_user = UserUser(15, min_nbrs=5)
algo = Recommender.adapt(user_user)
algo.fit(data_set.ratings)

swapnil_recommender = algo.recommend(-1, num_represent, ratings=pd.Series(swapnil_ratings_dict))
joined_data = swapnil_recommender.join(data_set.movies["genres"], on="item")
joined_data = joined_data.join(data_set.movies["title"], on="item")
joined_data = joined_data[joined_data.columns[2:]]
print("Recommendation For Swapnil: ")
print("")
print(joined_data.to_string())
print("")
print("---------------------------------------------------------------------------------------------------------------")

# By using the user-user collaborator algorithm, we did the same, but this time
# give a personal movie recommendation for my friend

my_friend_recommender = algo.recommend(-1, num_represent, ratings=pd.Series(my_friend_ratings_dict))
joined_data = my_friend_recommender.join(data_set.movies["genres"], on="item")
joined_data = joined_data.join(data_set.movies["title"], on="item")
joined_data = joined_data[joined_data.columns[2:]]
print("Recommendation For My Friend: ")
print("")
print(joined_data.to_string())
print("")
print("---------------------------------------------------------------------------------------------------------------")

# We are combining the movies we both watched from our dict and then storing an average of their
# ratings in a a combined dict

combined_rating_dict = {}

for k in swapnil_ratings_dict:
    if k in my_friend_ratings_dict:
        combined_rating_dict.update({k: float((swapnil_ratings_dict[k] + my_friend_ratings_dict[k])/2)})
    else:
        combined_rating_dict.update({k: swapnil_ratings_dict[k]})

for k in my_friend_ratings_dict:
    if k in swapnil_ratings_dict:
        combined_rating_dict.update({k: float((my_friend_ratings_dict[k] + swapnil_ratings_dict[k])/2)})
    else:
        combined_rating_dict.update({k: my_friend_ratings_dict[k]})

# We use the User-User collaborator yet again, but this time on our combined movie ratings dict
# to give a recommendation of movies for ourself, we can watch on a movie night

combined_recommender = algo.recommend(-1, num_represent, ratings=pd.Series(combined_rating_dict))
joined_data = combined_recommender.join(data_set.movies["genres"], on="item")
joined_data = joined_data.join(data_set.movies["title"], on="item")
joined_data = joined_data[joined_data.columns[2:]]
print("Recommendation Of Movies For Both Swapnil And His Friend: ")
print("")
print(joined_data.to_string())
print("")
print("---------------------------------------------------------------------------------------------------------------")
print("")
print("Thank you for using our recommender system, happy watching..................")
