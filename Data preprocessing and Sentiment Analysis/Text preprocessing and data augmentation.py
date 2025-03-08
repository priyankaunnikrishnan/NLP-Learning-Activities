import os
import pandas as pd
import re
import random
import nltk
import nlpaug.augmenter.word as naw
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import nltk
nltk.download('wordnet')
nltk.download('omw-1.4') 

# Download necessary NLP resources
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# STEP 1: Load the dataset (Choose the correct file based on your first name)
# Get the current script's directory
base_dir = os.path.dirname(os.path.abspath(__file__))
#prent directory
parent_dir = os.path.dirname(base_dir)
# Define the relative path
file_path = os.path.join(parent_dir, "Artificial_Intelligence_mini.csv")

print(" Loading dataset:", file_path)
priyanka_df = pd.read_csv(file_path)

# Display dataset info
print("\n Initial Data Exploration:")
print(priyanka_df.head())  # Show first few rows
print("\n Dataset Shape (Before Preprocessing):", priyanka_df.shape)

# STEP 2: Drop the 'user' column
print("\n Dropping unnecessary columns...")
priyanka_df.drop(columns=['user'], inplace=True)

# STEP 3: Clean text (remove retweets, mentions, hashtags, special characters)
def clean_text(text):
    text = re.sub(r'RT @\w+: ', '', text)  # Remove retweet markers
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Apply Lemmatization & Stopword Removal
    return ' '.join(words)

print("\n Cleaning tweets...")
priyanka_df['text'] = priyanka_df['text'].apply(clean_text)

# STEP 4: Load Word2Vec model
word2vec_model_path = os.environ.get("W2V_MODEL_PATH")
print("\n Loading Word2Vec model from:", word2vec_model_path)
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
print("Word2Vec model loaded successfully!")

# STEP 5: Apply Word Embedding Augmentation (Using Word2Vec)
word2vec_aug = naw.WordEmbsAug(
    model_type='word2vec',
    model_path=word2vec_model_path,
    action="substitute"
)

print("\n Applying Word2Vec-based augmentation...")
priyanka_df_after_word_augmenter = priyanka_df.copy()
priyanka_df_after_word_augmenter['text'] = priyanka_df_after_word_augmenter['text'].apply(word2vec_aug.augment)

# STEP 6: Combine original and augmented datasets
priyanka_df_after_word_augmenter = pd.concat([priyanka_df, priyanka_df_after_word_augmenter], ignore_index=True)

# Save the word-augmented dataset
word_augmented_file = "priyanka_df_after_word_augmenter.csv"
priyanka_df_after_word_augmenter.to_csv(word_augmented_file, index=False)
print(f" Word embedding augmentation completed! Saved to: {word_augmented_file}")
print("\n Dataset Shape (After Word Augmentation):", priyanka_df_after_word_augmenter.shape)
print("\n \Dataframe after word augmentationn\n", priyanka_df_after_word_augmenter.head())

# STEP 7: Apply Random Insertion Augmentation
print("\n Applying Random Insertion Augmentation...")
def random_insertion(text):
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords

    if len(words) < 2:
        return text  

    random_words = random.sample(words, 2)  

    for word in random_words:
        if word in word2vec_model:
            synonyms = word2vec_model.most_similar(word, topn=5)  # Get top 5 synonyms
            if synonyms:
                synonym = synonyms[0][0]  # Choose the most similar word
                text = text.replace(word, synonym, 1)  # Replace only one occurrence

    return text

priyanka_df_after_random_insertion = priyanka_df.copy()
priyanka_df_after_random_insertion['text'] = priyanka_df_after_random_insertion['text'].apply(random_insertion)

# STEP 8: Combine datasets (original + random insertion)
priyanka_df_after_random_insertion = pd.concat([priyanka_df, priyanka_df_after_random_insertion], ignore_index=True)

# Save the final dataset
random_inserted_file = "priyanka_df_after_random_insertion.txt"
priyanka_df_after_random_insertion.to_csv(random_inserted_file, index=False, sep='\t')
print(f" Random insertion augmentation completed! Saved to: {random_inserted_file}")
print(priyanka_df_after_random_insertion.head())
print("\nDataset Shape (After Random Insertion Augmentation):", priyanka_df_after_random_insertion.shape)

