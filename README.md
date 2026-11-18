# SentimentScope AI

> A clean and simple **Sentiment Analysis** project for product/movie reviews.  
> Perfect fresher-level AI project that is easy to explain and defend in interviews.

---

## 🎯 Problem

Companies receive thousands of customer reviews. Manually reading them is time-consuming.  
**SentimentScope AI** automatically classifies any review as **Positive**, **Negative**, or **Neutral**.

---

## ✨ Features

- Paste any review text → Instant sentiment prediction
- Shows confidence score
- Clean Streamlit web interface
- Word-level insights
- Fully explainable ML pipeline (no black-box deep learning)

---

## 🛠️ Tech Stack

| Component       | Technology              |
|----------------|-------------------------|
| Language       | Python                  |
| ML Library     | Scikit-learn            |
| Vectorizer     | TF-IDF                  |
| Model          | Logistic Regression     |
| Text Cleaning  | NLTK                    |
| Frontend       | Streamlit               |

---

## 🚀 How to Run

```bash
# 1. Clone / Extract the project
cd sentimentscope-ai

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data (first time only)
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

# 5. Train the model (run only once)
python app/train.py

# 6. Start the app
streamlit run app/main.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
sentimentscope-ai/
├── app/
│   ├── main.py           # Streamlit UI
│   ├── train.py          # Model training
│   ├── preprocess.py     # Text cleaning
│   └── predict.py        # Prediction logic
├── data/
│   └── sample_reviews.csv
├── models/               # Saved model goes here
├── tests/
│   └── test_preprocess.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🧠 How the Model Works

1. **Text Cleaning**
   - Lowercasing
   - Remove URLs, punctuation, numbers
   - Remove stopwords

2. **Feature Extraction**
   - TF-IDF Vectorizer (converts text → numbers)

3. **Classification**
   - Logistic Regression predicts: Positive / Negative / Neutral

4. **Output**
   - Sentiment label + Confidence percentage

This is a classic, industry-standard NLP pipeline that is very easy to explain.

---

## 🎤 How to Defend This Project in Interview

**Q. Why Logistic Regression instead of BERT or LSTM?**  
→ For a fresher project, interpretability and simplicity matter more. I can explain every step clearly. Deep models are harder to debug and need more data/compute.

**Q. What is TF-IDF and why did you use it?**  
→ TF-IDF gives higher importance to rare but meaningful words and lower importance to common words. It works very well with traditional ML models on text data.

**Q. What are the limitations of your project?**  
→ Limited training data, does not understand sarcasm or complex context very well, and performance depends on the quality of training examples.

**Q. How would you improve it?**  
→ Use a larger dataset, try transformer models (DistilBERT), add explainability (which words influenced the prediction), and deploy it as an API.

**Q. Real-world use case?**  
→ E-commerce companies, movie platforms, and customer support teams use sentiment analysis to understand customer feedback at scale.

---

## 📊 Sample Results

| Review | Prediction |
|--------|------------|
| "This product is amazing, totally worth the money!" | Positive |
| "Worst experience ever. Do not buy this." | Negative |
| "It's okay, nothing special." | Neutral |

---

## 👨‍💻 Author

Built as a portfolio project for AI / ML / Data roles.

**Feel free to fork and improve!**
