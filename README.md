# Document-summarization-tool

# Document Summarization Tool

This Python project provides a tool for summarizing documents in various formats (PDF, DOCX, TXT) using a combination of Hidden Markov Models (HMM) and a transformer-based summarization model. The tool first selects relevant sentences using HMM and then summarizes the selected sentences using a pre-trained transformer model.

## Features

- Extracts text from PDF, DOCX, and TXT files.
- Preprocesses text into sentences.
- Uses HMM to score and select the most relevant sentences.
- Summarizes the selected sentences using a transformer model.
- Supports summarization of multi-page documents.

## Requirements

- Python 3.7 or higher
- PyTorch or TensorFlow (for transformer models)
- `transformers` library
- `hmmlearn` library
- `sklearn` library
- `nltk` library
- `PyPDF2` library
- `python-docx` library

## Installation

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/document-summarizer.git
    cd document-summarizer
    ```

2. Create a virtual environment and activate it (optional but recommended):

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required libraries:

    ```bash
    pip install torch torchvision torchaudio  # For PyTorch
    # or
    pip install tensorflow  # For TensorFlow

    pip install transformers hmmlearn scikit-learn nltk PyPDF2 python-docx
    ```

4. Download the required NLTK data:

    ```python
    import nltk
    nltk.download('punkt')
    ```

## Usage

1. Place your document file (PDF, DOCX, or TXT) in the project directory or specify the full path.

2. Run the script:

    ```bash
    python document_summarizer.py
    ```

3. The script will extract text from the provided file, preprocess it, score sentences using HMM, and generate a summary. The output summary will be printed to the console.

### Example

To summarize a PDF file:

```bash
python document_summarizer.py
