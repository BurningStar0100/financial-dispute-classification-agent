print("starting import")

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import joblib
from datetime import timedelta
import logging

print("import completed")

# --- 1. Load Saved Models and Preprocessing Objects ---
def load_models():
    """
    Loads the trained model and associated artifacts from disk.
    """

    try:
        model = joblib.load('model/dispute_classifier_model.pkl')
        tfidf_vectorizer = joblib.load('model/tfidf_vectorizer.pkl')
        label_encoder = joblib.load('model/label_encoder.pkl')
        engineered_feature_names = joblib.load('model/engineered_features.pkl')
        print("Models and artifacts loaded successfully.")
        return model, tfidf_vectorizer, label_encoder, engineered_feature_names
    except FileNotFoundError:
        print("Error: Model files not found. Please run the training script first.")
        exit()




def preprocess_text(text):
    """Cleans and prepares text data for vectorization."""
    # Set up the same preprocessing functions as in training
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text, flags=re.I)
    text = text.lower()
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# --- 2. Real Feature Engineering for New Data ---

def perform_feature_engineering(disputes_df, transactions_df):
    """
    Creates the engineered feature columns for new data by looking up
    information in the transactions table.
    """
    # Ensure timestamp columns are in datetime format for comparison
    transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
    # Merge disputes with their primary transaction details
    merged_df = pd.merge(disputes_df, transactions_df, on='txn_id', how='left', suffixes=('', '_trans'))

    # --- Verification Logic ---

    # 2.1: Verify 'is_verified_failed'
    # Check the status of the transaction associated with the dispute.
    failed_statuses = ['FAILED', 'CANCELLED', 'CANCEL'] # Cover variations
    merged_df['is_verified_failed'] = merged_df['status'].isin(failed_statuses).astype(int)

    # 2.2: Verify 'is_verified_duplicate'
    # This is more complex: for each dispute, search for a similar transaction.
    def find_duplicate(row):
        customer_id = row['customer_id']
        amount = row['amount']
        dispute_time = row['timestamp']
        txn_id = row['txn_id']
        time_window = timedelta(minutes=3)

        # Search for another successful transaction from the same customer for the same amount
        # within a 5-minute window of the original transaction.
        potential_duplicates = transactions_df[
            (transactions_df['customer_id'] == customer_id) &
            (transactions_df['amount'] == amount) &
            (transactions_df['status'] == 'SUCCESS') &
            (transactions_df['txn_id'] != txn_id) &
            (transactions_df['timestamp'] >= dispute_time - time_window) &
            (transactions_df['timestamp'] <= dispute_time + time_window)
        ]
        return 1 if not potential_duplicates.empty else 0

    merged_df['is_verified_duplicate'] = merged_df.apply(find_duplicate, axis=1)

    # 2.3: Keyword Features (these remain the same)
    merged_df['contains_fraud_keyword'] = merged_df['description'].str.contains('fraud|unauthorized|suspicious', case=False, regex=True).astype(int)
    merged_df['contains_refund_keyword'] = merged_df['description'].str.contains('refund|waiting|canceled|debited|debit', case=False, regex=True).astype(int)
    merged_df['contains_duplicate_keyword'] = merged_df['description'].str.contains('twice|duplicate|double|two', case=False, regex=True).astype(int)

    return merged_df

# --- 3. Inference and Explanation Generation ---

def generate_explanation(row, predicted_class, feature_names, model, tfidf_vectorizer, label_encoder, engineered_feature_names ,top_n=3 ):
    """Generates a human-readable explanation for a prediction."""
    class_index = list(label_encoder.classes_).index(predicted_class)

    if len(model.coef_) == 1:
        coefs = model.coef_[0]
    else:
        coefs = model.coef_[class_index]

    feature_importance = pd.DataFrame({'feature': feature_names, 'importance': coefs})
    active_feature_indices = [i for i, val in enumerate(row) if val > 0]
    active_features = feature_importance.iloc[active_feature_indices]
    active_features['abs_importance'] = active_features['importance'].abs()
    top_features = active_features.sort_values(by='abs_importance', ascending=False).head(top_n)

    explanation_parts = []
    for _, feat_row in top_features.iterrows():
        feature_name = feat_row['feature']
        if feature_name in engineered_feature_names:
            explanation_parts.append(f"the rule '{feature_name.replace('_', ' ')}' was triggered")
        else:
            try:
                feature_index = int(feature_name)
                word = tfidf_vectorizer.get_feature_names_out()[feature_index]
                explanation_parts.append(f"the keyword '{word}'")
            except (ValueError, IndexError):
                continue

    if not explanation_parts:
        return "Prediction based on the overall text content."

    return f"Prediction primarily based on: {', '.join(explanation_parts)}."

# --- 4. Main Execution Block ---
def main():
    model, tfidf_vectorizer, label_encoder, engineered_feature_names = load_models()
    logging.basicConfig(level=logging.INFO,filename='dispute_classification.log', filemode='a',
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Load the unclassified disputes and the full transactions history
    try:
        # Use the 'disputes.csv' file as the input for unclassified data
        new_disputes_df = pd.read_csv('dataset/disputes.csv')
        transactions_df = pd.read_csv('dataset/transactions.csv')
        logging.info("Dispute and transaction data loaded.")
        # print("Dispute and transaction data loaded.")
    except FileNotFoundError as e:
        logging.error(f"Error loading data files: {e}")
        # print(f"Error: {e}. Please ensure 'disputes.csv' and 'transactions.csv' are in the same directory.")
        exit()

    # Apply text preprocessing
    new_disputes_df['processed_description'] = new_disputes_df['description'].apply(preprocess_text)

    # Apply the new, data-driven feature engineering
    logging.info("Performing feature engineering using transaction data...")
    print("Performing feature engineering using transaction data...")
    featured_disputes_df = perform_feature_engineering(new_disputes_df, transactions_df)
    
    logging.info("Feature engineering complete.")

    # Vectorize the text data
    X_new_tfidf = tfidf_vectorizer.transform(featured_disputes_df['processed_description'])

    # Combine with engineered features
    X_new_engineered = featured_disputes_df[engineered_feature_names]
    X_new_combined = pd.concat([pd.DataFrame(X_new_tfidf.toarray()), X_new_engineered.reset_index(drop=True)], axis=1)
    X_new_combined.columns = X_new_combined.columns.astype(str)

    # Make predictions
    predictions_encoded = model.predict(X_new_combined)
    probabilities = model.predict_proba(X_new_combined)

    # Decode predictions back to original labels
    predicted_categories = label_encoder.inverse_transform(predictions_encoded)
    confidence_scores = probabilities.max(axis=1)

    # Generate explanations
    explanations = [generate_explanation(X_new_combined.iloc[i].values, predicted_categories[i], X_new_combined.columns,model, tfidf_vectorizer, label_encoder, engineered_feature_names) for i in range(len(X_new_combined))]

    # Create the final output DataFrame
    output_df = pd.DataFrame({
        'dispute_id': featured_disputes_df['dispute_id'],
        'predicted_category': predicted_categories,
        'confidence': confidence_scores,
        'explanation': explanations,
        'contains_fraud_keyword': featured_disputes_df['contains_fraud_keyword'],
        "contains_refund_keyword": featured_disputes_df['contains_refund_keyword'],
        'is_verified_duplicate': featured_disputes_df['is_verified_duplicate'],
        'is_verified_failed': featured_disputes_df['is_verified_failed'],
    })

    # Save the results to a CSV file
    output_df.to_csv('results/classified_disputes.csv', index=False , mode = "w")

    logging.info("\nInference complete. Results saved to 'classified_disputes.csv'")
    print(output_df)

