import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import argparse
import os
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class NetflixRecommendationSystem:
    def __init__(self):
        self.ratings_df = None
        self.movies_df = None
        self.user_item_matrix = None
        self.user_similarity = None
        self.item_similarity = None
        self.trained = False
        
    def load_data(self, ratings_path: str, movies_path: str = None):
        """Load movie ratings and movie metadata"""
        print("Loading data...")
        
        # Load ratings data
        self.ratings_df = pd.read_csv(ratings_path)
        print(f"Loaded {len(self.ratings_df)} ratings")
        
        # Load movies data if provided
        if movies_path and os.path.exists(movies_path):
            self.movies_df = pd.read_csv(movies_path)
            print(f"Loaded {len(self.movies_df)} movies")
        else:
            # Create basic movie info from ratings if movies file not available
            self.movies_df = pd.DataFrame({
                'movieId': self.ratings_df['movieId'].unique(),
                'title': [f"Movie_{mid}" for mid in self.ratings_df['movieId'].unique()]
            })
            print("Created basic movie metadata from ratings")
    
    def preprocess_data(self):
        """Clean and preprocess the data"""
        print("Preprocessing data...")
        
        # Remove users with very few ratings (less than 5)
        user_counts = self.ratings_df['userId'].value_counts()
        active_users = user_counts[user_counts >= 5].index
        self.ratings_df = self.ratings_df[self.ratings_df['userId'].isin(active_users)]
        
        # Remove movies with very few ratings (less than 10)
        movie_counts = self.ratings_df['movieId'].value_counts()
        popular_movies = movie_counts[movie_counts >= 10].index
        self.ratings_df = self.ratings_df[self.ratings_df['movieId'].isin(popular_movies)]
        
        print(f"After preprocessing: {len(self.ratings_df)} ratings, "
              f"{self.ratings_df['userId'].nunique()} users, "
              f"{self.ratings_df['movieId'].nunique()} movies")
    
    def create_user_item_matrix(self):
        """Create user-item rating matrix"""
        print("Creating user-item matrix...")
        self.user_item_matrix = self.ratings_df.pivot_table(
            index='userId', 
            columns='movieId', 
            values='rating'
        ).fillna(0)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
    
    def calculate_similarities(self):
        """Calculate user-user and item-item similarities"""
        print("Calculating similarities...")
        
        # User-based collaborative filtering
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_similarity = pd.DataFrame(
            self.user_similarity,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.index
        )
        
        # Item-based collaborative filtering
        self.item_similarity = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            self.item_similarity,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )
        
        print("Similarity matrices calculated")
    
    def train(self, ratings_path: str, movies_path: str = None):
        """Train the recommendation system"""
        self.load_data(ratings_path, movies_path)
        self.preprocess_data()
        self.create_user_item_matrix()
        self.calculate_similarities()
        self.trained = True
        print("Training completed!")
    
    def get_user_recommendations(self, user_id: int, n_recommendations: int = 10, 
                               method: str = 'user_based') -> List[Tuple[int, str, float]]:
        """Get movie recommendations for a user"""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        if user_id not in self.user_item_matrix.index:
            return self._get_popular_movies(n_recommendations)
        
        if method == 'user_based':
            return self._user_based_recommendations(user_id, n_recommendations)
        elif method == 'item_based':
            return self._item_based_recommendations(user_id, n_recommendations)
        else:
            # Hybrid approach - average of both methods
            user_recs = self._user_based_recommendations(user_id, n_recommendations * 2)
            item_recs = self._item_based_recommendations(user_id, n_recommendations * 2)
            
            # Combine and rank
            combined_scores = {}
            for movie_id, title, score in user_recs:
                combined_scores[movie_id] = {'title': title, 'score': score}
            
            for movie_id, title, score in item_recs:
                if movie_id in combined_scores:
                    combined_scores[movie_id]['score'] = (combined_scores[movie_id]['score'] + score) / 2
                else:
                    combined_scores[movie_id] = {'title': title, 'score': score}
            
            # Sort by combined score
            sorted_recs = sorted(combined_scores.items(), key=lambda x: x[1]['score'], reverse=True)
            return [(mid, data['title'], data['score']) for mid, data in sorted_recs[:n_recommendations]]
    
    def _user_based_recommendations(self, user_id: int, n_recommendations: int) -> List[Tuple[int, str, float]]:
        """Generate recommendations using user-based collaborative filtering"""
        user_ratings = self.user_item_matrix.loc[user_id]
        user_similarities = self.user_similarity.loc[user_id].sort_values(ascending=False)
        
        # Get top similar users (excluding the user itself)
        similar_users = user_similarities.iloc[1:51].index  # Top 50 similar users
        
        recommendations = {}
        for movie_id in self.user_item_matrix.columns:
            if user_ratings[movie_id] == 0:  # User hasn't rated this movie
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_user in similar_users:
                    if self.user_item_matrix.loc[similar_user, movie_id] > 0:
                        similarity = user_similarities[similar_user]
                        rating = self.user_item_matrix.loc[similar_user, movie_id]
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[movie_id] = predicted_rating
        
        # Sort recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get movie titles
        result = []
        for movie_id, score in sorted_recs[:n_recommendations]:
            title = self._get_movie_title(movie_id)
            result.append((movie_id, title, score))
        
        return result
    
    def _item_based_recommendations(self, user_id: int, n_recommendations: int) -> List[Tuple[int, str, float]]:
        """Generate recommendations using item-based collaborative filtering"""
        user_ratings = self.user_item_matrix.loc[user_id]
        rated_movies = user_ratings[user_ratings > 0].index
        
        recommendations = {}
        for movie_id in self.user_item_matrix.columns:
            if user_ratings[movie_id] == 0:  # User hasn't rated this movie
                weighted_sum = 0
                similarity_sum = 0
                
                for rated_movie in rated_movies:
                    similarity = self.item_similarity.loc[movie_id, rated_movie]
                    if similarity > 0:
                        rating = user_ratings[rated_movie]
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    recommendations[movie_id] = predicted_rating
        
        # Sort recommendations
        sorted_recs = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        
        # Get movie titles
        result = []
        for movie_id, score in sorted_recs[:n_recommendations]:
            title = self._get_movie_title(movie_id)
            result.append((movie_id, title, score))
        
        return result
    
    def _get_popular_movies(self, n_recommendations: int) -> List[Tuple[int, str, float]]:
        """Get popular movies for new users"""
        movie_popularity = self.ratings_df.groupby('movieId').agg({
            'rating': ['mean', 'count']
        }).round(2)
        movie_popularity.columns = ['avg_rating', 'rating_count']
        
        # Weight by both average rating and popularity
        movie_popularity['score'] = (
            movie_popularity['avg_rating'] * 0.7 + 
            (movie_popularity['rating_count'] / movie_popularity['rating_count'].max()) * 0.3 * 5
        )
        
        top_movies = movie_popularity.sort_values('score', ascending=False).head(n_recommendations)
        
        result = []
        for movie_id, row in top_movies.iterrows():
            title = self._get_movie_title(movie_id)
            result.append((movie_id, title, row['score']))
        
        return result
    
    def _get_movie_title(self, movie_id: int) -> str:
        """Get movie title by ID"""
        movie_row = self.movies_df[self.movies_df['movieId'] == movie_id]
        if not movie_row.empty:
            return movie_row.iloc[0]['title']
        return f"Movie_{movie_id}"
    
    def evaluate_model(self, test_size: float = 0.2) -> Dict[str, float]:
        """Evaluate the recommendation system using RMSE and MAE"""
        if not self.trained:
            raise ValueError("Model not trained. Call train() first.")
        
        print("Evaluating model...")
        
        # Split data
        train_data, test_data = train_test_split(self.ratings_df, test_size=test_size, random_state=42)
        
        # Create training matrix
        train_matrix = train_data.pivot_table(
            index='userId', columns='movieId', values='rating'
        ).fillna(0)
        
        # Calculate similarities on training data
        train_user_similarity = cosine_similarity(train_matrix)
        train_user_similarity = pd.DataFrame(
            train_user_similarity,
            index=train_matrix.index,
            columns=train_matrix.index
        )
        
        predictions = []
        actuals = []
        
        for _, row in test_data.iterrows():
            user_id, movie_id, actual_rating = row['userId'], row['movieId'], row['rating']
            
            if user_id in train_matrix.index and movie_id in train_matrix.columns:
                # Predict rating using user-based CF
                user_similarities = train_user_similarity.loc[user_id].sort_values(ascending=False)
                similar_users = user_similarities.iloc[1:21].index  # Top 20 similar users
                
                weighted_sum = 0
                similarity_sum = 0
                
                for similar_user in similar_users:
                    if train_matrix.loc[similar_user, movie_id] > 0:
                        similarity = user_similarities[similar_user]
                        rating = train_matrix.loc[similar_user, movie_id]
                        weighted_sum += similarity * rating
                        similarity_sum += abs(similarity)
                
                if similarity_sum > 0:
                    predicted_rating = weighted_sum / similarity_sum
                    predictions.append(predicted_rating)
                    actuals.append(actual_rating)
        
        if predictions:
            rmse = np.sqrt(mean_squared_error(actuals, predictions))
            mae = mean_absolute_error(actuals, predictions)
            
            return {
                'RMSE': round(rmse, 4),
                'MAE': round(mae, 4),
                'Predictions': len(predictions)
            }
        else:
            return {'Error': 'No predictions could be made'}
    
    def get_user_stats(self, user_id: int) -> Dict:
        """Get statistics for a specific user"""
        if user_id not in self.ratings_df['userId'].values:
            return {'Error': 'User not found'}
        
        user_ratings = self.ratings_df[self.ratings_df['userId'] == user_id]
        
        return {
            'Total Ratings': len(user_ratings),
            'Average Rating': round(user_ratings['rating'].mean(), 2),
            'Rating Distribution': user_ratings['rating'].value_counts().to_dict(),
            'Favorite Genres': self._get_user_favorite_genres(user_id)
        }
    
    def _get_user_favorite_genres(self, user_id: int) -> str:
        """Get user's favorite genres (simplified)"""
        # This would require genre information in the movies dataset
        return "Feature requires genre data in movies dataset"


class NetflixCLI:
    def __init__(self):
        self.recommender = NetflixRecommendationSystem()
    
    def run(self):
        """Run the CLI interface"""
        parser = argparse.ArgumentParser(description='Netflix Movie Recommendation System')
        parser.add_argument('--ratings', required=True, help='Path to ratings CSV file')
        parser.add_argument('--movies', help='Path to movies CSV file (optional)')
        parser.add_argument('--train', action='store_true', help='Train the model')
        parser.add_argument('--recommend', type=int, help='Get recommendations for user ID')
        parser.add_argument('--method', choices=['user_based', 'item_based', 'hybrid'], 
                          default='hybrid', help='Recommendation method')
        parser.add_argument('--num', type=int, default=10, help='Number of recommendations')
        parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
        parser.add_argument('--stats', type=int, help='Get user statistics')
        parser.add_argument('--interactive', action='store_true', help='Run interactive mode')
        
        args = parser.parse_args()
        
        if args.train or args.recommend or args.evaluate or args.stats or args.interactive:
            print("🎬 Netflix Recommendation System")
            print("=" * 50)
            self.recommender.train(args.ratings, args.movies)
        
        if args.recommend:
            print(f"\n🎯 Recommendations for User {args.recommend} ({args.method}):")
            print("-" * 50)
            recommendations = self.recommender.get_user_recommendations(
                args.recommend, args.num, args.method
            )
            
            for i, (movie_id, title, score) in enumerate(recommendations, 1):
                print(f"{i:2d}. {title:<40} (Score: {score:.3f})")
        
        if args.evaluate:
            print("\n📊 Model Evaluation:")
            print("-" * 30)
            metrics = self.recommender.evaluate_model()
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
        
        if args.stats:
            print(f"\n👤 User {args.stats} Statistics:")
            print("-" * 30)
            stats = self.recommender.get_user_stats(args.stats)
            for key, value in stats.items():
                print(f"{key}: {value}")
        
        if args.interactive:
            self.interactive_mode()
    
    def interactive_mode(self):
        """Run interactive CLI mode"""
        print("\n🎮 Interactive Mode - Enter commands:")
        print("Commands: recommend <user_id>, stats <user_id>, evaluate, quit")
        
        while True:
            try:
                command = input("\n> ").strip().lower().split()
                
                if not command:
                    continue
                
                if command[0] == 'quit':
                    print("Goodbye! 👋")
                    break
                
                elif command[0] == 'recommend' and len(command) > 1:
                    user_id = int(command[1])
                    method = command[2] if len(command) > 2 else 'hybrid'
                    num_recs = int(command[3]) if len(command) > 3 else 10
                    
                    print(f"\n🎯 Recommendations for User {user_id}:")
                    recommendations = self.recommender.get_user_recommendations(
                        user_id, num_recs, method
                    )
                    
                    for i, (movie_id, title, score) in enumerate(recommendations, 1):
                        print(f"{i:2d}. {title:<40} (Score: {score:.3f})")
                
                elif command[0] == 'stats' and len(command) > 1:
                    user_id = int(command[1])
                    print(f"\n👤 User {user_id} Statistics:")
                    stats = self.recommender.get_user_stats(user_id)
                    for key, value in stats.items():
                        print(f"{key}: {value}")
                
                elif command[0] == 'evaluate':
                    print("\n📊 Evaluating model...")
                    metrics = self.recommender.evaluate_model()
                    for metric, value in metrics.items():
                        print(f"{metric}: {value}")
                
                else:
                    print("Invalid command. Try: recommend <user_id>, stats <user_id>, evaluate, quit")
            
            except (ValueError, IndexError, KeyError) as e:
                print(f"Error: {e}")
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break


# Example usage and testing
if __name__ == "__main__":
    # If running as script, use CLI
    cli = NetflixCLI()
    cli.run()
    
    # Example of programmatic usage:
    """
    # Initialize recommender
    recommender = NetflixRecommendationSystem()
    
    # Train the model
    recommender.train('ratings.csv', 'movies.csv')
    
    # Get recommendations
    recommendations = recommender.get_user_recommendations(user_id=123, n_recommendations=10)
    
    # Evaluate model
    metrics = recommender.evaluate_model()
    print(f"RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}")
    """