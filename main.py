from preprocessing import preprocess_main
from extract import extract_main
from evaluate import evaluate_main

if __name__ == "__main__":
    # Preprocess articles
    preprocess_main()

    # Extract 4W from articles
    extract_main()

    # Evaluate the extraction
    evaluate_main()
