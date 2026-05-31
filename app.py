import string
import pickle
import streamlit as st

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

ps = PorterStemmer()


def trans_text(text):
    text=text.lower()
    text=nltk.word_tokenize(text)
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text=y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))


st.title('📩 Email/SMS Spam Classifier')
st.caption('A lightweight Machine Learning app powered by Multinomial Naive Bayes to instantly detect spam.')
st.write("---") 


input_sms = st.text_area(
    "Enter the message to classify", 
    height=150, 
    placeholder="Type or paste your SMS or Email content here..."
)


col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button('🔍 Predict Message', use_container_width=True)

if predict_button:
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message to classify.")
    else:
        transformed_sms = trans_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        st.write("### Result:")

        if result == 1:
            st.error("**Spam Detected!** This message looks suspicious.")
        else:
            st.success("**Not Spam!** This message appears perfectly safe.")