from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import logging
import os
import gdown
import json
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rapidfuzz import fuzz, process   # ✅ added for fuzzy matching

logging.basicConfig(level=logging.INFO)

app = Flask(__name__, template_folder="templates", static_folder="static")



# ========== QUIZ QUESTIONS ==========
questions = [
    "🔧 Do you enjoy hands-on activities like fixing gadgets, repairing things, or working with mechanical tools?",
    "🌳 Do you prefer spending time outdoors and being physically active rather than sitting indoors for long hours?",
    "🧩 Do you like solving complex problems, puzzles, or understanding how things work at a deeper level?",
    "🔬 Are you interested in conducting experiments, doing research, or exploring abstract scientific ideas?",
    "🎨 Do you feel fulfilled when you express yourself creatively through art, music, writing, or design?",
    "🕒 Do you enjoy having flexibility in your work schedule rather than following a strict routine?",
    "❤️ Do you feel happy when helping, teaching, or taking care of others?",
    "👂 Are you good at listening to people and helping them resolve conflicts or problems?",
    "👔 Do you enjoy leading teams, persuading others, or taking on business challenges?",
    "🚀 Do you feel ambitious and motivated to take risks in order to achieve bigger goals?",
    "📋 Do you like working in an organized environment with clear rules, predictable tasks, and step-by-step processes?",
    "📊 Do you enjoy managing data, handling budgets, or keeping detailed records and reports?"
]

dimension_map = [
    "Realistic", "Realistic", "Investigative", "Investigative",
    "Artistic", "Artistic", "Social", "Social",
    "Enterprising", "Enterprising", "Conventional", "Conventional"
]

degree_map = {
    "Science": [("B.Tech in CS", "💻"), ("B.Sc Physics", "⚛️"), ("MBBS", "🩺"), ("BCA", "🖥")],
    "Commerce": [("B.Com Hons", "📚"), ("CA", "🧾"), ("BBA Finance", "💹"), ("BA Economics", "💵")],
    "Arts": [("BA Psychology", "🧠"), ("BFA", "🎨"), ("BA Journalism", "📰"), ("BA English Lit", "📖")],
    "Vocational": [("Diploma Web Designing", "💻"), ("ITI (Electrical)", "⚡"), ("B.Voc Hospitality", "🏨"), ("Skill Plumbing", "🔧")]
}

career_map = {
    "Science": [("Software Engineer","💻"),("Research Scientist","🔬"),("Doctor","🩺"),("Data Scientist","📊")],
    "Commerce": [("Accountant","📒"),("Investment Banker","💰"),("Entrepreneur","🚀"),("Financial Analyst","📈")],
    "Arts": [("Psychologist","🧠"),("Graphic Designer","🎨"),("Journalist","📰"),("Content Writer","✍️")],
    "Vocational": [("Full-Stack Dev","💻"),("Electrician","⚡"),("Hotel Manager","🏨"),("Mechanic","🔧")]
}

# ========== ROUTES ==========
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/mapping")
def path_mapping():
    return render_template("mapping.html")



@app.route("/college_map")
def college_map():
    return render_template("college_map.html")

@app.route("/mentor")
def mentor():
    return render_template("mentor.html")

@app.route("/exam_tracker")
def tracker():
    return render_template("exam_tracker.html")


@app.route("/quiz")
def quiz():
    return render_template("quiz.html", questions_json=questions)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_models()
        if model is None or label_encoder is None:
            return jsonify({'error': 'Model or label encoder not loaded on server. Check server logs.'}), 500

        body = request.get_json(force=True)
        answers = body.get('answers')

        if not isinstance(answers, list):
            return jsonify({'error': 'answers must be a list'}), 400
        if len(answers) != len(questions):
            return jsonify({'error': f'Provide {len(questions)} answer values (1-5). Received {len(answers)}.'}), 400

        answers_int = [int(x) for x in answers]
        X = np.array([answers_int])
        pred_encoded = model.predict(X)
        pred_text = label_encoder.inverse_transform(pred_encoded)[0]

        dimension_scores = {}
        for i, dim in enumerate(dimension_map):
            dimension_scores[dim] = dimension_scores.get(dim, 0) + int(answers_int[i])

        messages = {
            "Science":"🔬 Explore, experiment, and innovate!",
            "Commerce":"💰 Learn financial thinking and business basics.",
            "Arts":"🎨 Grow your creative and communication skills.",
            "Vocational":"🔧 Develop hands-on and employable skills."
        }

        careers = career_map.get(pred_text, [])
        degrees = degree_map.get(pred_text, [])

        response = {
            'recommendation': pred_text,
            'message': messages.get(pred_text, ''),
            'careers': careers,
            'degrees': degrees,
            'dimension_scores': dimension_scores
        }
        return jsonify(response)

    except Exception as e:
        logging.exception("Error in /predict")
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500





# ========== MAIN ==========
if __name__ == "__main__":
    app.run(debug=True)






























