import glob
import numpy as np
from nltk.corpus import names
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


# ==================== FUNCȚII UTILE ==================== #

def letters_only(word):
    return word.isalpha()

def clean_text(docs):
    clean_docs = []
    for doc in docs:
        clean_docs.append(' '.join([
            lemmatizer.lemmatize(word.lower())
            for word in doc.split()
            if letters_only(word) and word not in all_names
        ]))
    return clean_docs

def get_label_index(labels):
    from collections import defaultdict
    label_index = defaultdict(list)
    for index, label in enumerate(labels):
        label_index[label].append(index)
    return label_index

def get_prior(label_index):
    prior = {label: len(index) for label, index in label_index.items()}
    total_count = sum(prior.values())
    for label in prior:
        prior[label] /= float(total_count)
    return prior

def get_likelyhood(term_document_matrix, label_index, smoothing=1):
    likelihood = {}
    for label, index in label_index.items():
        likelihood[label] = term_document_matrix[index, :].sum(axis=0) + smoothing
        likelihood[label] = np.asarray(likelihood[label])[0].astype(float)
        total_count = likelihood[label].sum()
        likelihood[label] = likelihood[label] / total_count
    return likelihood

def get_posterior(term_document_matrix, prior, likelihood):
    num_docs = term_document_matrix.shape[0]
    posteriors = []

    for i in range(num_docs):
        posterior = {label: np.log(prior[label]) for label in prior}
        term_document_vector = term_document_matrix.getrow(i)
        counts = term_document_vector.data
        indices = term_document_vector.indices

        for label in likelihood:
            for count, index in zip(counts, indices):
                posterior[label] += np.log(likelihood[label][index]) * count

        min_log = min(posterior.values())
        for label in posterior:
            try:
                posterior[label] = np.exp(posterior[label] - min_log)
            except OverflowError:
                posterior[label] = float('inf')

        total = sum(posterior.values())
        for label in posterior:
            if posterior[label] == float('inf'):
                posterior[label] = 1.0
            else:
                posterior[label] /= total

        posteriors.append(posterior.copy())

    return posteriors

def predict(text):
    cleaned = clean_text([text])
    vectorized = cv.transform(cleaned)
    post = get_posterior(vectorized, prior, likelihood)[0]
    prediction = max(post, key=post.get)
    print("Spam probability:", post[1])
    print("Ham probability:", post[0])
    return "SPAM" if prediction == 1 else "HAM"

# ==================== INIȚIALIZARE ==================== #

all_names = set(names.words())
lemmatizer = WordNetLemmatizer()

emails, labels = [], []

for filename in glob.glob('enron1/spam/*.txt'):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(1)

for filename in glob.glob('enron1/ham/*.txt'):
    with open(filename, 'r', encoding="ISO-8859-1") as infile:
        emails.append(infile.read())
        labels.append(0)

print(f"Loaded {len(emails)} emails.")

# ==================== PRE-PROCESARE TEXT ==================== #

cleaned_emails = clean_text(emails)

cv = CountVectorizer(stop_words="english", max_features=500)
X_train, X_test, Y_train, Y_test = train_test_split(cleaned_emails, labels, test_size=0.33, random_state=42)
print(len(X_train), len(X_test), len(Y_train), len(Y_test))

term_docs_train = cv.fit_transform(X_train)
label_index = get_label_index(Y_train)
prior = get_prior(label_index)
likelihood = get_likelyhood(term_docs_train, label_index, smoothing=1)

term_docs_test = cv.transform(X_test)
posterior = get_posterior(term_docs_test, prior, likelihood)

# ==================== EVALUARE MODEL ==================== #

predicted_labels = [max(p, key=p.get) for p in posterior]
correct = sum(1 for pred, actual in zip(predicted_labels, Y_test) if pred == actual)
accuracy = correct / len(Y_test) * 100
print(f"The accuracy on {len(Y_test)} testing samples is: {accuracy:.1f}%")

# ==================== TEST MANUAL ==================== #

sample = "Congratulations! You won a free ticket to Bahamas. Claim now!"
print("Predicție pentru emailul tău:", predict(sample))
