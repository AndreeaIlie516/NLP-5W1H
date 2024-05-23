import pandas as pd
from sklearn.metrics import classification_report


def evaluate_4w(true_df, pred_df):
    metrics = ["who", "what", "when", "where"]
    results = {}
    for metric in metrics:
        y_true = true_df[metric].apply(lambda x: x.split(", ") if isinstance(x, str) else [])
        y_pred = pred_df[metric].apply(lambda x: x.split(", ") if isinstance(x, str) else [])

        true_positives = [len(set(a).intersection(set(b))) for a, b in zip(y_true, y_pred)]
        false_positives = [len(set(b) - set(a)) for a, b in zip(y_true, y_pred)]
        false_negatives = [len(set(a) - set(b)) for a, b in zip(y_true, y_pred)]

        precision = sum(true_positives) / (sum(true_positives) + sum(false_positives) + 1e-6)
        recall = sum(true_positives) / (sum(true_positives) + sum(false_negatives) + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)

        results[metric] = {"precision": precision, "recall": recall, "f1": f1}

    return pd.DataFrame(results).T


def evaluate_main():
    # Load extracted articles
    true_df = pd.read_excel("data/test_articles_processed.xlsx")
    pred_df = pd.read_excel("data/test_articles_extracted.xlsx")

    # Evaluate
    evaluation_results = evaluate_4w(true_df, pred_df)
    print("Evaluation results:")
    print(evaluation_results)


if __name__ == "__main__":
    evaluate_main()
