# Netflix-Recommendation-System

A machine learning-based movie recommendation system implementing collaborative filtering techniques to suggest personalized movie recommendations based on user ratings and viewing history.

## Features

- **Collaborative Filtering**: Implements both user-based and item-based collaborative filtering algorithms
- **Hybrid Recommendations**: Combines multiple recommendation approaches for improved accuracy
- **CLI Interface**: Interactive command-line interface for easy user interaction
- **Performance Evaluation**: Includes RMSE and MAE metrics to assess algorithm performance
- **Data Processing**: Comprehensive data preprocessing and analysis using Pandas and NumPy

## Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms and metrics
- **Cosine Similarity** - For calculating user and item similarities

##  How It Works

### 1. Data Processing
- Loads movie ratings and metadata from CSV files
- Cleans data by removing users with few ratings and unpopular movies
- Creates user-item rating matrices for analysis

### 2. Collaborative Filtering
- **User-Based**: Finds users with similar preferences and recommends movies they liked
- **Item-Based**: Recommends movies similar to ones the user has already rated highly
- **Hybrid Approach**: Combines both methods for more accurate recommendations

### 3. Similarity Calculation
Uses cosine similarity to measure relationships between:
- Users (based on their rating patterns)
- Movies (based on how users rate them)

## � Installation & Usage

### Prerequisites
```bash
pip install pandas numpy scikit-learn
```

### Quick Start
1. Clone the repository:
```bash
git clone https://github.com/ShuricorE/Netflix-Recommendation-System.git
cd Netflix-Recommendation-System
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Generate sample data:
```bash
python3 create_sample_data.py
```

4. Get movie recommendations:
```bash
python3 main.py --ratings ratings_small.csv --movies movies_small.csv --train --recommend 1
```

##  CLI Commands

### Basic Usage
```bash
# Get recommendations for user ID 1
python3 main.py --ratings ratings_small.csv --movies movies_small.csv --train --recommend 1

# Get 5 recommendations using user-based method
python3 main.py --ratings ratings_small.csv --movies movies_small.csv --train --recommend 1 --method user_based --num 5

# Evaluate model performance
python3 main.py --ratings ratings_small.csv --movies movies_small.csv --train --evaluate

# Get user statistics
python3 main.py --ratings ratings_small.csv --movies movies_small.csv --train --stats 1
```

### Interactive Mode
```bash
python3 main.py --ratings ratings_small.csv --movies movies_small.csv --train --interactive
```

Interactive commands:
- `recommend <user_id>` - Get recommendations
- `stats <user_id>` - View user statistics
- `evaluate` - Run model evaluation
- `quit` - Exit

##  Sample Output

```
🎬 Netflix Recommendation System
==================================================
Loading data...
Loaded 584 ratings
Loaded 20 movies
Preprocessing data...
After preprocessing: 584 ratings, 50 users, 20 movies
Creating user-item matrix...
User-item matrix shape: (50, 20)
Calculating similarities...
Similarity matrices calculated
Training completed!

 Recommendations for User 1 (hybrid):
--------------------------------------------------
 1. Interstellar (2014)                    (Score: 3.844)
 2. The Godfather (1972)                   (Score: 3.705)
 3. The Matrix (1999)                      (Score: 3.735)
 4. Toy Story (1995)                       (Score: 3.691)
 5. Star Wars: A New Hope (1977)           (Score: 3.691)
```

##  Performance Metrics

The system includes evaluation metrics to assess recommendation quality:
- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average prediction error
- **Coverage**: Percentage of items the system can recommend

##  Configuration Options

### Recommendation Methods
- `user_based`: Recommends based on similar users
- `item_based`: Recommends based on similar movies
- `hybrid`: Combines both approaches (default)

### Data Requirements
Your CSV files should have these columns:
- **ratings.csv**: `userId`, `movieId`, `rating`, `timestamp`
- **movies.csv**: `movieId`, `title`, `genres` (optional)

##  Project Structure

```
Netflix-Recommendation-System/
├── main.py                 # Main recommendation system
├── create_sample_data.py   # Generates sample dataset
├── ratings_small.csv       # Sample ratings data
├── movies_small.csv        # Sample movie metadata
├── requirements.txt        # Python dependencies
└── README.md              # Project documentation
```

##  Key Algorithms

### User-Based Collaborative Filtering
1. Calculate similarity between users using cosine similarity
2. Find top-N most similar users
3. Predict ratings based on similar users' preferences
4. Recommend highest-predicted unrated movies

### Item-Based Collaborative Filtering
1. Calculate similarity between movies
2. For each unrated movie, find similar movies the user has rated
3. Predict rating based on similar movies' ratings
4. Recommend movies with highest predicted ratings

## 🔍 Core Competencies Demonstrated

- **Machine Learning**: Implementation of collaborative filtering algorithms
- **Data Preprocessing**: Data cleaning, normalization, and matrix operations
- **Statistical Analysis**: Similarity calculations and performance evaluation
- **Software Engineering**: Modular code design and CLI development
- **Python Programming**: Advanced use of Pandas, NumPy, and Scikit-learn

##  Future Enhancements

- [ ] Content-based filtering using movie genres
- [ ] Deep learning approaches (Neural Collaborative Filtering)
- [ ] Real-time recommendation updates
- [ ] Web interface using Flask/Django
- [ ] Integration with movie databases (TMDB API)
- [ ] Handling cold start problem for new users

##  Contributing

Feel free to fork this project and submit pull requests for improvements!

##  License

This project is open source and available under the [MIT License](LICENSE).

##  Author

**Nidhi Prasad**
- Email: ndhprd03@gmail.com
- GitHub: [@ShuricorE](https://github.com/ShuricorE)

---

*Built as part of a machine learning portfolio to demonstrate collaborative filtering and recommendation system implementation.*
