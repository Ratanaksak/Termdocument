from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
import os
from collections import Counter

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload_and_display.html', bow=None)

@app.route('/process_file', methods=['POST'])
def process_file():
    # Get the uploaded file
    uploaded_file = request.files['file']

    if uploaded_file.filename != '':
        # Save the file temporarily
        file_path = os.path.join('uploads', uploaded_file.filename)
        uploaded_file.save(file_path)

        # Process the file and generate Bag-of-Words
        bow = generate_bow(file_path)

        # Remove the temporary file
        os.remove(file_path)

        return render_template('upload_and_display.html', bow=bow)
    else:
        return render_template('upload_and_display.html', bow=None)

def generate_bow(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Split the content into documents (assuming one document per line)
    documents = content.split('\n')

    # Initialize the CountVectorizer
    vectorizer = CountVectorizer()

    # Fit and transform the documents to get the Bag-of-Words (Term-Document Matrix)
    bow_matrix = vectorizer.fit_transform(documents).toarray()

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    # Calculate word frequencies
    word_frequencies = Counter({feature_names[i]: int(bow_matrix[:, i].sum()) for i in range(bow_matrix.shape[1])})

    return word_frequencies

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    os.makedirs('uploads', exist_ok=True)

    # Run the Flask application
    app.run(debug=True)
