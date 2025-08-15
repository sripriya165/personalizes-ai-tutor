from flask import Flask, request, render_template
from transformers import pipeline

app = Flask(__name__)

# Load a small pre-trained model for Q&A
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

context = """
Python is a high-level, interpreted programming language known for its simplicity and versatility. 
It is widely used in data science, AI, web development, automation, and many other fields. 
AI, or Artificial Intelligence, refers to the simulation of human intelligence by machines.
"""

@app.route("/", methods=["GET", "POST"])
def home():
    response = ""
    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if question:
            try:
                answer = qa_pipeline(question=question, context=context)
                response = answer['answer']
            except Exception as e:
                response = "Sorry, I couldn't process that question."
    return render_template("index.html", response=response)

if __name__ == "__main__":
    app.run(debug=True)
