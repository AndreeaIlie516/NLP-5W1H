import spacy
import pandas as pd

# Load Romanian model with dependency parser and NER
nlp = spacy.load('ro_core_news_sm')

# List of common temporal words in Romanian (with and without diacritics)
temporal_words = ["luni", "marti", "miercuri", "joi", "vineri", "sambata", "duminica",
                  "ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie",
                  "august", "septembrie", "octombrie", "noiembrie", "decembrie",
                  "astazi", "ieri", "maine", "saptamana", "luna", "anul",
                  "luni", "marți", "miercuri", "joi", "vineri", "sâmbătă", "duminică",
                  "ianuarie", "februarie", "martie", "aprilie", "mai", "iunie", "iulie",
                  "august", "septembrie", "octombrie", "noiembrie", "decembrie",
                  "astăzi", "ieri", "mâine", "săptămâna", "luna", "anul"]


def extract_who(doc):
    persons = []
    for token in doc:
        if token.dep_ in ['nsubj', 'dobj', 'iobj', 'appos'] and token.ent_type_ == 'PERSON':
            print(f"WHO: {token.text} - {token.dep_} - {token.ent_type_}")
            persons.append(token.text)
        elif token.head.dep_ in ['nsubj', 'dobj', 'iobj'] and token.head.ent_type_ == 'PERSON':
            print(f"WHO (head): {token.head.text} - {token.head.dep_} - {token.head.ent_type_}")
            persons.append(token.head.text)
    return list(set(persons))


def extract_what(doc):
    actions = []
    for token in doc:
        if token.dep_ == 'ROOT':
            action = token.text
            for child in token.children:
                if child.dep_ == 'dobj':
                    action += " " + child.text
            print(f"WHAT: {action}")
            actions.append(action)
    return actions


def extract_when(doc):
    times = []
    for token in doc:
        if token.dep_ in ['advmod', 'tmod'] or token.ent_type_ in ['DATE',
                                                                   'TIME'] or token.text.lower() in temporal_words:
            print(f"WHEN: {token.text} - {token.dep_} - {token.ent_type_}")
            times.append(token.text)
    return list(set(times))


def extract_where(doc):
    locations = []
    for token in doc:
        if token.ent_type_ in ['GPE', 'LOC']:
            if token.dep_ in ['obl', 'nmod'] or (token.head.dep_ == 'prep' and token.head.head.pos_ == 'VERB'):
                print(f"WHERE: {token.text} - {token.dep_} - {token.ent_type_}")
                locations.append(token.text)
    return list(set(locations))


def extract_4w_from_sentence(sentence):
    doc = nlp(sentence)
    for token in doc:
        print(f"Token: {token.text} - Dep: {token.dep_} - Ent: {token.ent_type_} - Head: {token.head.text}")
    return {
        "who": extract_who(doc),
        "what": extract_what(doc),
        "when": extract_when(doc),
        "where": extract_where(doc),
    }


def extract_4w(article):
    sentences = [sent.text for sent in nlp(article).sents]
    who, what, when, where = [], [], [], []
    for sentence in sentences:
        result = extract_4w_from_sentence(sentence)
        who.extend(result['who'])
        what.extend(result['what'])
        when.extend(result['when'])
        where.extend(result['where'])
    return {
        "who": list(set(who)),
        "what": list(set(what)),
        "when": list(set(when)),
        "where": list(set(where)),
    }


def extract_main():
    # Load processed articles
    train_df = pd.read_excel("data/train_articles_processed.xlsx")
    test_df = pd.read_excel("data/test_articles_processed.xlsx")

    # Apply extraction
    train_df['extracted'] = train_df['article'].apply(extract_4w)
    test_df['extracted'] = test_df['article'].apply(extract_4w)

    # Convert the extracted results from dictionaries to separate columns
    train_extracted_df = pd.DataFrame(train_df['extracted'].tolist())
    test_extracted_df = pd.DataFrame(test_df['extracted'].tolist())

    # Concatenate the extracted results with the original data
    train_df = pd.concat([train_df, train_extracted_df], axis=1)
    test_df = pd.concat([test_df, test_extracted_df], axis=1)

    # Drop the intermediate 'extracted' column
    train_df = train_df.drop(columns=['extracted'])
    test_df = test_df.drop(columns=['extracted'])

    # Save the extracted results
    train_df.to_excel("data/train_articles_extracted.xlsx", index=False)
    test_df.to_excel("data/test_articles_extracted.xlsx", index=False)
    print("4W extraction completed and saved.")

    # Example test
    text = ("Președintele Klaus Iohannis a declarat marți la București că România va susține Ucraina. "
            "Ministrul Educației, Sorin Cîmpeanu, a prezentat ieri noul plan de reformă al sistemului școlar.")
    results = extract_4w(text)
    for key, value in results.items():
        print(f"{key.capitalize()}: {', '.join(value)}")


if __name__ == "__main__":
    extract_main()
