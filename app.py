from flask import Flask, request, render_template, redirect, url_for, jsonify 
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import nltk
from nltk.corpus import stopwords
import os
import base64

nltk.download("stopwords")
STOPWORDS = set(stopwords.words("english"))

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def preprocess_text(text):
    return " ".join(word for word in text.lower().split() if word not in STOPWORDS)

def extract_text_from_pdf(filepath):
    text = ""
    with open(filepath, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_title = request.form["job_title"]
        job_description = request.form["job_description"]
        preprocessed_description = preprocess_text(job_description)
        uploaded_files = request.files.getlist("resumes")

        resume_texts = []
        resume_files = []
        for file in uploaded_files:
            if file and file.filename.endswith('.pdf'):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                content = extract_text_from_pdf(filepath)
                preprocessed_content = preprocess_text(content)
                resume_texts.append(preprocessed_content)
                resume_files.append(filename)

        if not resume_texts:
            return redirect(url_for("index"))

        all_texts = [preprocessed_description] + resume_texts
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(all_texts)
        cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

        ranked_resumes = sorted(zip(resume_files, cosine_similarities), key=lambda x: x[1], reverse=True)

        return render_template("results.html", job_title=job_title, ranked_resumes=ranked_resumes)

    return render_template("index.html")

@app.route("/api/rank_resumes", methods=["POST"])
def rank_resumes():
    data = request.get_json()
    job_title = data.get("job_title")
    job_description = data.get("job_description")
    preprocessed_description = preprocess_text(job_description)
    
    resume_texts = []
    resume_files = []

    for resume in data.get("resumes", []):
        filename = resume.get("filename")
        file_content = resume.get("content")

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, "wb") as file:
            file.write(base64.b64decode(file_content))

        content = extract_text_from_pdf(filepath)
        preprocessed_content = preprocess_text(content)
        resume_texts.append(preprocessed_content)
        resume_files.append(filename)

    all_texts = [preprocessed_description] + resume_texts
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(all_texts)
    cosine_similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    ranked_resumes = sorted(zip(resume_files, cosine_similarities), key=lambda x: x[1], reverse=True)

    return jsonify({
        "job_title": job_title,
        "ranked_resumes": [{"resume": filename, "score": round(score * 100, 2)} for filename, score in ranked_resumes]
    })

if __name__ == "__main__":
    app.run(debug=True)
