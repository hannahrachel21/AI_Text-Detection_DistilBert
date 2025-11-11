import pandas as pd
import string
import logging
import os
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_preprocessing_log.txt")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_path, mode='w', encoding='utf-8')
    ]
)

#Load and view the dataset
df1 = pd.read_csv('/mnt/c/ai_detection/ai_dataset/research-abstracts-labeled.csv')
df2 = pd.read_csv('/mnt/c/ai_detection/ai_dataset/wiki-labeled.csv')

#Combine the dataframes
combined_df = pd.concat([df1, df2], ignore_index=True)

#Filter based on min word count - to not feed too short sentences to the model
min_word_count = 30
filtered_df = combined_df[combined_df['word_count'] >= min_word_count].reset_index(drop=True)
logging.info(f"'research-abstracts-labeled.csv' size: {df1.shape}")
logging.info(f"'wiki-labeled.csv' size: {df2.shape}")
logging.info(f"Combined dataframe size: {combined_df.shape}")
logging.info(f"Filtered dataframe after removing rows with <{min_word_count} word count: {filtered_df.shape}")

#Drop columns - title and word_count
filtered_df = filtered_df.drop(['title', 'word_count'], axis=1)

#Drop missing values
filtered_df.dropna(subset=['text', 'label'], inplace=True)

#Lowercasing
filtered_df['text'] = filtered_df['text'].str.lower()

#Remove html tags
filtered_df['text'] = filtered_df['text'].str.replace(r'<.*?>', '', regex=True)

#Remove url tags
filtered_df['text'] = filtered_df['text'].str.replace(r'https?://\S+|www\.\S+', '', regex=True)

#Remove punctuation and special characters
filtered_df['text'] = filtered_df['text'].str.translate(str.maketrans('', '', string.punctuation))

logging.info("The dataframe is lowercased, and clean of html tags, urls, punctuations, and special characters")
logging.info(f"Cleaned dataframe size: {filtered_df.shape}")
logging.info(f"Cleaned dataframe columns: {filtered_df.columns}")
logging.info(f"Labels: {filtered_df['label'].unique()}")
logging.info(F"No. of unique labels in new dataset: {filtered_df['label'].value_counts()}")

#Save the cleaned dataframe to csv
filtered_df.to_csv("/mnt/c/ai_detection/ai_dataset/ai_human.csv")
logging.info("Saved as 'ai_human.csv'")