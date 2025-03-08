import pandas as pd
import re
import nltk
from sklearn.metrics import accuracy_score, f1_score
import os

# Download stopwords for further improvements if needed
nltk.download('stopwords')
 
# STEP 1: Load the dataset (Choose the correct file based on your first name)
# Get the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))
#prent directory
parent_dir = os.path.dirname(base_dir)
# Define the relative path
file_path = os.path.join(parent_dir, "Artificial_intelligence_data.csv")
print(f"\n Loading dataset: {file_path}")

try:
    # Try reading the file while handling bad rows
    priyanka_df = pd.read_csv(file_path, sep=",", engine="python", on_bad_lines="skip")
except Exception as e:
    print(f" Error loading dataset: {e}")
    exit()

# Display dataset info
print("\n Initial Data Exploration:")
print(priyanka_df.head())
print("\n Dataset Shape (Before Preprocessing):", priyanka_df.shape)

# STEP 2: Drop 'user' column (if it exists)
if 'user' in priyanka_df.columns:
    print("\n Dropping unnecessary columns...")
    priyanka_df.drop(columns=['user'], inplace=True)

# STEP 3: Clean tweets (remove retweets, mentions, hashtags, special characters)
def clean_text(text):
    text = str(text)  # Ensure text is string
    text = re.sub(r'RT @\w+: ', '', text)  # Remove retweet markers
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    return text

print("\n Cleaning tweets...")
priyanka_df['text'] = priyanka_df['text'].apply(clean_text)

# STEP 4: Handle Negations
def handle_negation(text):
    negation_words = {"not", "never", "no", "n't", "none", "nobody", "nothing", "nowhere", "neither", "hardly", "scarcely", "barely"}
    words = text.split()
    new_words = []
    negate = False

    for word in words:
        if word in negation_words:
            negate = True
        elif negate:
            word = "NOT_" + word
            negate = False
        new_words.append(word)
    
    return ' '.join(new_words)

priyanka_df['text'] = priyanka_df['text'].apply(handle_negation)

# STEP 5: Add tweet length column
priyanka_df['tweet_len'] = priyanka_df['text'].apply(lambda x: len(x.split()))
print("\n Added 'tweet_len' column!")

print(priyanka_df.head())

# STEP 6: Load Positive and Negative Lexicons
pos_lexicon_path = os.path.join(parent_dir, "positive-words.txt")
neg_lexicon_path = os.path.join(parent_dir, "negative-words.txt")
try:
    print("\n Loading lexicons with ISO-8859-1 encoding...")
    positive_words = set(pd.read_csv(pos_lexicon_path, header=None, encoding='ISO-8859-1')[0])
    negative_words = set(pd.read_csv(neg_lexicon_path, header=None, encoding='ISO-8859-1')[0])
    print(" Lexicons loaded successfully!")
except Exception as e:
    print(f"\nError loading lexicons: {e}")
    positive_words, negative_words = set(), set()

# STEP 7: Compute Positivity and Negativity Percentages
def calculate_sentiment(text, word_count):
    words = text.split()
    pos_hits = sum(1 for word in words if word in positive_words)
    neg_hits = sum(1 for word in words if word in negative_words)
    
    pos_percentage = (pos_hits / word_count) if word_count > 0 else 0
    neg_percentage = (neg_hits / word_count) if word_count > 0 else 0
    
    return pos_percentage, neg_percentage

print("\n Calculating sentiment scores...")
priyanka_df[['pos_percentage', 'neg_percentage']] = priyanka_df.apply(
    lambda row: calculate_sentiment(row['text'], row['tweet_len']), axis=1, result_type='expand')

# STEP 8: Assign Predicted Sentiments
def assign_sentiment(pos, neg):
    if pos == neg or (pos == 0 and neg == 0):
        return "neutral"
    elif pos > neg:
        return "positive"
    else:
        return "negative"

priyanka_df['predicted_sentiment_score'] = priyanka_df.apply(
    lambda row: assign_sentiment(row['pos_percentage'], row['neg_percentage']), axis=1)

print("\n Predicted sentiment scores assigned!")

# STEP 9: Compare with Actual Sentiment and Calculate Accuracy & F1 Score
if 'sentiment' in priyanka_df.columns:
    actual_sentiments = priyanka_df['sentiment']
    predicted_sentiments = priyanka_df['predicted_sentiment_score']

    accuracy = accuracy_score(actual_sentiments, predicted_sentiments)
    f1 = f1_score(actual_sentiments, predicted_sentiments, average='weighted')

    print(f"\n Accuracy: {accuracy:.2f}")
    print(f" F1 Score: {f1:.2f}")
else:
    print("\n 'sentiment' column not found. Cannot compute Accuracy & F1 Score.")

# Save final dataframe
output_file = os.path.join(parent_dir, "priyanka_sentiment_analysis_results.csv")
priyanka_df.to_csv(output_file, index=False)
print(f"\n Sentiment analysis completed! Results saved to: {output_file}")
