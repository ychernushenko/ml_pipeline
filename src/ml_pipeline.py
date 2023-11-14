import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def load_data(file_path, column_names):
    """
    Load data from TSV files into a pandas DataFrame.
    """
    try:
        return pd.read_csv(file_path, encoding='utf8', sep='\t', names=column_names)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def correct_labels_spelling(products_df, label_col_name, mapping=None):
    """
    Correct misspelled labels according to 'mapping'.
    """
    for typo, correct_name in mapping.items():
        products_df.loc[products_df[label_col_name] == typo] = correct_name

def preprocess_data(data_root_dir):
    """
    Preprocess product and review data.
    """
    product_files = [f'{data_root_dir}/products-data-{i}.tsv' for i in range(4)]

    # Some of the files have different order of columns
    review_files_order_1 = [f'{data_root_dir}/reviews-{i}.tsv' for i in range(4) if i != 2]
    review_files_order_2 = [f'{data_root_dir}/reviews-{i}.tsv' for i in [2]]

    products_df = pd.concat([load_data(f, ["id", "category", "product_title"]) for f in product_files], ignore_index=True)
    correct_labels_spelling(products_df, "category", {"Ktchen": "Kitchen"})
    reviews_df_order_1 = pd.concat([load_data(f, ["id", "rating", "review_text"]) for f in review_files_order_1], ignore_index=True)
    reviews_df_order_2 = pd.concat([load_data(f, ["rating", "id", "review_text"]) for f in review_files_order_2], ignore_index=True)
    reviews_df = pd.concat([reviews_df_order_1, reviews_df_order_2], ignore_index=True)

    return pd.merge(products_df, reviews_df, on="id")

def build_model():
    """
    Build a machine learning pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('review_text', TfidfVectorizer(), 'review_text'),
            ('product_title', TfidfVectorizer(), 'product_title'),
            ('rating', StandardScaler(), ['rating'])
        ],
        remainder='passthrough'
    )

    return Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression())
    ])

def main(data_root_dir):    
    df = preprocess_data(data_root_dir)

    features = ['review_text', 'product_title', 'rating']
    target = 'category'

    X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')

    print('\nClassification Report:')
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and evaluate a model on the provided dataset.')
    parser.add_argument('--data_root_dir', type=str, required=True, help='Root directory of the dataset.')

    args = parser.parse_args()
    main(data_root_dir=args.data_root_dir)