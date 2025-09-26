from inference.dispute_classification import main as classify_disputes
from inference.resolution_suggestion import main as suggest_resolutions
from inference.query_cli import main as query_cli
# --- 1. Dispute Classification ---
if __name__ == "__main__":
    print("Starting Dispute Classification...")
    classify_disputes()
    print("Dispute Classification Completed.\n")

    # --- 2. Resolution Suggestion ---
    print("Starting Resolution Suggestion...")
    suggest_resolutions()
    print("Resolution Suggestion Completed.\n")

    # --- 3. Query CLI ---
    print("Starting Query CLI...")
    query_cli()