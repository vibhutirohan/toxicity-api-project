import pandas as pd
import requests
import time
from collections import Counter

# ============================================
# CHANGE THIS TO YOUR ACTUAL VERCEL API URL
# ============================================
API_URL = "https://toxicity-api-project.vercel.app/predict"

# ============================================
# DATASET FILE
# ============================================
DATASET_PATH = "language_detection_dataset.csv"

# ============================================
# OPTIONAL SETTINGS
# ============================================
CONNECT_TIMEOUT = 5
READ_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS = 0.3   # small pause to avoid hitting too fast

def main():
    # Load dataset
    df = pd.read_csv(DATASET_PATH)

    # Basic validation
    required_columns = ["text", "label"]
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column in dataset: {col}")

    print(f"Loaded dataset: {DATASET_PATH}")
    print(f"Total rows found: {len(df)}")

    results = []
    status_counter = Counter()
    actual_counter = Counter()
    predicted_counter = Counter()

    # Optional warm-up request
    print("\nSending warm-up request to wake up Vercel...")
    try:
        warmup_payload = {"message": "Hello, this is a warm-up request."}
        warmup_response = requests.post(
            API_URL,
            json=warmup_payload,
            timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
        )
        print(f"Warm-up status: {warmup_response.status_code}")
    except Exception as e:
        print(f"Warm-up failed: {e}")

    print("\nStarting full dataset test...\n")

    for i, row in df.iterrows():
        message_text = str(row["text"]).strip()
        actual_label = str(row["label"]).strip().upper()

        actual_counter[actual_label] += 1

        payload = {
            "message": message_text
        }

        # If your API expects sender also, use this instead:
        # payload = {
        #     "sender": "test-user",
        #     "message": message_text
        # }

        try:
            response = requests.post(
                API_URL,
                json=payload,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT)
            )

            status_code = response.status_code
            status_counter[str(status_code)] += 1

            if status_code == 200:
                data = response.json()

                # Change "label" below if your API returns a different key
                predicted_label = str(data.get("label", "")).strip().upper()
                predicted_counter[predicted_label] += 1

                is_correct = predicted_label == actual_label

                results.append({
                    "row_number": i + 1,
                    "text": message_text,
                    "actual_label": actual_label,
                    "predicted_label": predicted_label,
                    "status_code": status_code,
                    "correct": is_correct,
                    "raw_response": str(data)
                })
            else:
                results.append({
                    "row_number": i + 1,
                    "text": message_text,
                    "actual_label": actual_label,
                    "predicted_label": None,
                    "status_code": status_code,
                    "correct": False,
                    "raw_response": response.text
                })

        except requests.exceptions.ConnectTimeout:
            status_counter["CONNECT_TIMEOUT"] += 1
            results.append({
                "row_number": i + 1,
                "text": message_text,
                "actual_label": actual_label,
                "predicted_label": None,
                "status_code": "CONNECT_TIMEOUT",
                "correct": False,
                "raw_response": "Connection timeout"
            })

        except requests.exceptions.ReadTimeout:
            status_counter["READ_TIMEOUT"] += 1
            results.append({
                "row_number": i + 1,
                "text": message_text,
                "actual_label": actual_label,
                "predicted_label": None,
                "status_code": "READ_TIMEOUT",
                "correct": False,
                "raw_response": "Read timeout"
            })

        except requests.exceptions.RequestException as e:
            status_counter["REQUEST_ERROR"] += 1
            results.append({
                "row_number": i + 1,
                "text": message_text,
                "actual_label": actual_label,
                "predicted_label": None,
                "status_code": "REQUEST_ERROR",
                "correct": False,
                "raw_response": str(e)
            })

        # Progress log
        if (i + 1) % 10 == 0 or (i + 1) == len(df):
            print(f"Processed {i + 1}/{len(df)} rows")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Save detailed results
    results_df.to_csv("api_test_results.csv", index=False)

    # Only successful API calls
    success_df = results_df[results_df["status_code"] == 200]

    # Accuracy
    if len(success_df) > 0:
        accuracy = (success_df["correct"].mean()) * 100
    else:
        accuracy = 0.0

    # Per-class accuracy
    class_summary = []
    for label in sorted(results_df["actual_label"].dropna().unique()):
        label_df = success_df[success_df["actual_label"] == label]
        total = len(label_df)
        correct = int(label_df["correct"].sum()) if total > 0 else 0
        class_acc = (correct / total * 100) if total > 0 else 0.0

        class_summary.append({
            "label": label,
            "tested_rows_with_200": total,
            "correct_predictions": correct,
            "accuracy_percent": round(class_acc, 2)
        })

    class_summary_df = pd.DataFrame(class_summary)
    class_summary_df.to_csv("class_summary.csv", index=False)

    # Simple confusion matrix
    if len(success_df) > 0:
        confusion = pd.crosstab(
            success_df["actual_label"],
            success_df["predicted_label"],
            rownames=["Actual"],
            colnames=["Predicted"],
            dropna=False
        )
        confusion.to_csv("confusion_matrix.csv")
    else:
        confusion = pd.DataFrame()

    # Print final summary
    print("\n========== FINAL SUMMARY ==========")
    print(f"Total dataset rows           : {len(df)}")
    print(f"Successful 200 responses     : {len(success_df)}")
    print(f"Failed / non-200 responses   : {len(results_df) - len(success_df)}")
    print(f"Overall accuracy (200 only)  : {accuracy:.2f}%")

    print("\nStatus Code Summary:")
    for key, value in status_counter.items():
        print(f"  {key}: {value}")

    print("\nActual Label Distribution:")
    for key, value in actual_counter.items():
        print(f"  {key}: {value}")

    print("\nPredicted Label Distribution:")
    for key, value in predicted_counter.items():
        print(f"  {key}: {value}")

    if not class_summary_df.empty:
        print("\nPer-Class Accuracy:")
        print(class_summary_df.to_string(index=False))

    if not confusion.empty:
        print("\nConfusion Matrix:")
        print(confusion)

    print("\nSaved files:")
    print("  - api_test_results.csv")
    print("  - class_summary.csv")
    print("  - confusion_matrix.csv")

if __name__ == "__main__":
    main()