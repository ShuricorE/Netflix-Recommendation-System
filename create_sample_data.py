import pandas as pd
import numpy as np
import random

# Set random seed for reproducible results
np.random.seed(42)
random.seed(42)

# Create sample movie data
movies_data = {
    'movieId': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'title': [
        'The Dark Knight (2008)',
        'Inception (2010)',
        'Pulp Fiction (1994)',
        'The Matrix (1999)',
        'Forrest Gump (1994)',
        'The Godfather (1972)',
        'Interstellar (2014)',
        'Fight Club (1999)',
        'Goodfellas (1990)',
        'The Shawshank Redemption (1994)',
        'Avengers: Endgame (2019)',
        'Titanic (1997)',
        'Avatar (2009)',
        'Star Wars: A New Hope (1977)',
        'Jurassic Park (1993)',
        'The Lion King (1994)',
        'Toy Story (1995)',
        'Finding Nemo (2003)',
        'The Incredibles (2004)',
        'WALL-E (2008)'
    ]
}

# Create sample ratings data
ratings_data = []
user_ids = range(1, 51)  # 50 users
movie_ids = range(1, 21)  # 20 movies

for user_id in user_ids:
    # Each user rates 8-15 random movies
    num_ratings = random.randint(8, 15)
    rated_movies = random.sample(list(movie_ids), num_ratings)
    
    for movie_id in rated_movies:
        # Generate realistic ratings (more 3s, 4s, 5s than 1s, 2s)
        rating = np.random.choice([1, 2, 3, 4, 5], p=[0.05, 0.1, 0.25, 0.35, 0.25])
        timestamp = random.randint(1000000000, 1600000000)  # Random timestamp
        
        ratings_data.append({
            'userId': user_id,
            'movieId': movie_id,
            'rating': rating,
            'timestamp': timestamp
        })

# Create DataFrames
movies_df = pd.DataFrame(movies_data)
ratings_df = pd.DataFrame(ratings_data)

# Save to CSV files
movies_df.to_csv('movies_small.csv', index=False)
ratings_df.to_csv('ratings_small.csv', index=False)

print("✅ Created sample data files:")
print(f"📊 movies_small.csv: {len(movies_df)} movies")
print(f"⭐ ratings_small.csv: {len(ratings_df)} ratings")
print(f"👥 Users: {ratings_df['userId'].nunique()}")
print(f"🎬 Movies: {ratings_df['movieId'].nunique()}")

print("\n📈 Sample ratings distribution:")
print(ratings_df['rating'].value_counts().sort_index())

print("\n🎯 Now run:")
print("python3 main.py --ratings ratings_small.csv --movies movies_small.csv --train --recommend 1")