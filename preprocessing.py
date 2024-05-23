import pandas as pd
import os
import json


def load_json_files(directory):
    articles = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    if 'content' in item:  # Adjust this key based on the actual structure
                        articles.append({'article': item['content']})
                    elif 'text' in item:  # Try different keys based on the structure
                        articles.append({'article': item['text']})
                    elif 'body' in item:  # Add more keys as necessary
                        articles.append({'article': item['body']})
    return pd.DataFrame(articles)


def preprocess_articles(df):
    if 'article' not in df.columns:
        raise KeyError("The DataFrame does not contain the 'article' column.")
    df['id'] = range(1, len(df) + 1)
    df_processed = df[['id', 'article']].copy()
    return df_processed


def preprocess_main():
    # Load the dataset from JSON files
    df = load_json_files("datasets")

    # Preprocess the articles
    df_processed = preprocess_articles(df)

    # Split into training and test sets
    train_df = df_processed.sample(frac=0.8, random_state=42)
    test_df = df_processed.drop(train_df.index)

    # Save to Excel files
    train_df.to_excel("data/train_articles_old.xlsx", index=False)
    test_df.to_excel("data/test_articles_old.xlsx", index=False)

    print("Training and test articles preprocessed and saved.")


if __name__ == "__main__":
    preprocess_main()
