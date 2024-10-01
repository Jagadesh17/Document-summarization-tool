from flask import Flask, request, render_template
from document_summarizer import summarize_document, summarize_text  # Import summarization functions
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        file_path = os.path.join('uploads', file.filename)
        file.save(file_path)
        summary = summarize_document(file_path)
        return render_template('result.html', summary=summary)
    
    elif 'text' in request.form and request.form['text'].strip() != '':
        input_text = request.form['text']
        summary = summarize_text(input_text)
        return render_template('result.html', summary=summary)
    
    return "Please upload a file or enter some text."

if __name__ == "__main__":
    app.run(debug=True)
