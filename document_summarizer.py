import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from hmmlearn import hmm
from nltk.tokenize import sent_tokenize
from PyPDF2 import PdfReader
from docx import Document

def extract_text_from_pdf(pdf_path):
    text = ''
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def extract_text_from_docx(docx_path):
    text = ''
    doc = Document(docx_path)
    for para in doc.paragraphs:
        text += para.text + '\n'
    return text

def extract_text_from_txt(txt_path):
    with open(txt_path, 'r') as file:
        return file.read()

def preprocess_text(text):
    sentences = sent_tokenize(text)
    return sentences

def vectorize_sentences(sentences):
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(sentences)
    return X, vectorizer

def build_hmm_model(X):
    # Create and train the HMM model
    model = hmm.GaussianHMM(n_components=5, covariance_type='diag', n_iter=1000, random_state=42)
    model.fit(X.toarray())
    return model

def score_sentences(sentences, model, X):
    # Score each sentence using the HMM model
    _, frame_log_likelihoods = model.score_samples(X.toarray())
    
    # Average the log likelihoods for each sentence
    avg_log_likelihoods = np.mean(frame_log_likelihoods, axis=1)
    return list(zip(sentences, avg_log_likelihoods))

def summarize_text(text, summarizer, max_length=150, min_length=30):
    # Split text into chunks if it's too long
    chunks = [text[i:i + 1000] for i in range(0, len(text), 1000)]
    
    summaries = []
    for chunk in chunks:
        # Adjust max_length based on the length of each chunk
        adjusted_max_length = min(max_length, max(len(chunk) // 2, min_length))
        summary = summarizer(chunk, max_length=adjusted_max_length, min_length=min_length)[0]['summary_text']
        summaries.append(summary)
    
    return ' '.join(summaries)



def main(file_path):
    # Determine file type
    if file_path.endswith('.pdf'):
        text = extract_text_from_pdf(file_path)
    elif file_path.endswith('.docx'):
        text = extract_text_from_docx(file_path)
    elif file_path.endswith('.txt'):
        text = extract_text_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format")

    # Preprocess and vectorize text
    sentences = preprocess_text(text)
    X, vectorizer = vectorize_sentences(sentences)

    # Build and apply HMM model
    model = build_hmm_model(X)
    scored_sentences = score_sentences(sentences, model, X)

    # Select top sentences based on HMM scores
    top_sentences = [sentence for sentence, score in sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:10]]

    # Summarize the selected sentences
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarize_text(' '.join(top_sentences), summarizer)

    return summary

if __name__ == "__main__":
    # Example usage
    file_path = 'D:/Projects/KEIS/r-programming.pdf'  # Replace with your file path
    summary = main(file_path)
    print("Summary:")
    print(summary)
