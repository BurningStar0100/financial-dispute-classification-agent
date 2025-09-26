import pandas as pd

# --- 1. Load Data ---
try:
    classified_df = pd.read_csv('results/classified_disputes.csv')
    disputes_df = pd.read_csv('dataset/disputes.csv')
    print("All necessary data files loaded successfully.")
except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure 'classified_disputes.csv' and 'disputes.csv' are present.")
    exit()

# --- 2. Merge Data Sources ---
# Combine the two dataframes on 'dispute_id' to get all necessary columns in one place.
# We only need 'txn_type' and 'amount' from the original disputes file.
merged_df = pd.merge(
    classified_df,
    disputes_df[['dispute_id', 'txn_type', 'amount']],
    on='dispute_id',
    how='left'
)
print("Data merged successfully. Shape of merged data:", merged_df.shape)


# --- 3. The Rule-Based Decision Engine ---

def suggest_resolution(row):
    """
    Applies a set of prioritized rules to suggest a next action and justification.
    """
    # Rule 1 (Highest Priority): Handle Fraud
    if row['predicted_category'] == 'FRAUD':
        return 'Mark as potential fraud', "Prediction indicates a high risk of fraud, requiring immediate security review."

    # Rule 2: Handle specific transaction types that need escalation
    if row['txn_type'] == 'NEFT':
        return 'Escalate to bank', "NEFT transactions require inter-bank communication and have longer settlement cycles."

    # Rule 3: Handle high-confidence, verified cases for auto-refund
    # Conditions for this rule are strict to ensure safety.
    is_verified_duplicate = row.get('is_verified_duplicate', 0) == 1
    is_verified_failed = row.get('is_verified_failed', 0) == 1
    is_high_confidence = row['confidence'] > 0.95
    is_low_amount = row['amount'] < 2000

    if is_high_confidence and is_low_amount:
        if row['predicted_category'] == 'DUPLICATE_CHARGE' and is_verified_duplicate:
            return 'Auto-refund', "High-confidence prediction validated by transaction data; amount is within auto-approval limit."
        if row['predicted_category'] == 'FAILED_TRANSACTION' and is_verified_failed:
            return 'Auto-refund', "High-confidence prediction of a failed but debited transaction; amount is within auto-approval limit."

    # Rule 4: Handle ambiguous 'OTHERS' category
    if row['predicted_category'] == 'OTHERS':
         return 'Ask for more info', "The system could not confidently determine the issue from the provided description. More details are needed."

    # Rule 5 (Default): Catch-all for manual review
    # We generate a more specific justification for why manual review is needed.
    justification = "Dispute requires agent intervention."
    if row['amount'] >= 2000:
        justification = "Transaction amount exceeds the auto-approval limit."
    elif row['predicted_category'] == 'REFUND_PENDING':
        justification = "Category 'REFUND_PENDING' requires an agent to check the merchant's refund status."
    elif (row['predicted_category'] == 'DUPLICATE_CHARGE' and not is_verified_duplicate):
        justification = "Claim of duplicate charge could not be verified in transaction logs."

    return 'Manual review', justification


# --- 4. Apply the Engine and Save Results ---

# Apply the function to each row of the DataFrame.
# The result of the apply function is a list of tuples, which we can split into two new columns.
print("Applying resolution suggestion engine...")
resolutions = merged_df.apply(suggest_resolution, axis=1, result_type='expand')
merged_df['suggested_action'] = resolutions[0]
merged_df['justification'] = resolutions[1]
print("Resolution suggestions generated.")

# Prepare the final output DataFrame as per Task 2 requirements
output_df = merged_df[['dispute_id', 'suggested_action', 'justification']]

# Save the results to a new CSV file
output_df.to_csv('results/resolutions.csv', index=False , mode= "w")

print("\nTask 2 complete. Results saved to 'resolutions.csv'")
print(output_df.head())
