# 📩 Email/SMS Spam Classifier

### 🌐 Live Web App: https://sms-span-classifier---machine-learning-project---scklearn---ap.streamlit.app/

An end-to-end Machine Learning pipeline designed to automatically classify text messages as either **Spam** or **Ham** (Legitimate). This project marks my very first journey into Machine Learning, covering the full lifecycle from raw data exploration to a live web application deployment!

---

## 🚀 Project Overview
The objective of this project is to build a reliable classifier that filters out spam messages using Natural Language Processing (NLP) and probabilistic modeling. The model is trained on the classic SMS Spam Collection dataset, optimized for high precision, and hosted on a public web interface where users can test text messages in real-time.

## 🛠 Tech Stack
* **Language:** Python
* **Machine Learning & NLP:** Scikit-learn, NLTK, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Framework:** Streamlit
* **Deployment Platform:** Streamlit Community Cloud

---

## ⚙️ Key Workflow Stages

1. **Data Cleaning:** Dropped unnecessary columns, renamed features, handled missing values, and encoded target variables (`0` for Ham, `1` for Spam).
2. **Exploratory Data Analysis (EDA):** Analyzed the imbalance between classes, visualized character/word/sentence counts, and generated custom Word Clouds to discover prominent spam keywords.
3. **Text Preprocessing (NLP Pipeline):** * Converted text to lowercase.
   * Tokenized sentences into individual words.
   * Filtered out special characters, punctuation, and English stop words.
   * Applied **Porter Stemming** to reduce words to their base forms (e.g., *running* $\rightarrow$ *run*).
4. **Model Building & Evaluation:** Vectorized text using `TfidfVectorizer` and compared multiple classification algorithms (including Random Forest, SVM, and Naive Bayes). Selected **Multinomial Naive Bayes** as the final architecture due to its exceptional precision score on text data.
5. **Web Application & Deployment:** Developed a user-friendly frontend using Streamlit and deployed it live to the cloud.

---

## 📂 Project Structure

```text
sms-spam-detector/
│
├── app.py               # Main Streamlit web application script
├── requirements.txt     # List of required production packages for the server
├── model.pkl            # Serialized Multinomial Naive Bayes classifier
├── vectorizer.pkl       # Serialized TfidfVectorizer preprocessing object
└── spam_sms_nb.ipynb    # Jupyter Notebook containing full EDA and model training
