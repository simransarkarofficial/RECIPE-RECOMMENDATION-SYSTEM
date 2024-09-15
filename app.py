import numpy as np
import pandas as pd
import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download nltk data files (only run once)
nltk.download('punkt')
nltk.download('stopwords')

# Load the data
food_dict = pickle.load(open('D:\\PROJECTS\\CHATBOT RECIPE RECOMMENDOR\\food.pkl', 'rb'))

food = pd.DataFrame(food_dict)

# Define CSS for custom styles
page_bg_css = """
<style>
body {
    background-color: #E7D4B5;
}
</style>
"""

st.markdown(page_bg_css, unsafe_allow_html=True)

custom_title_css = """
<style>
h1 {
    color: #ff6347;  /* Change this to your desired color */
}
</style>
"""

st.markdown(custom_title_css, unsafe_allow_html=True)

st.title("Recipe Recommendation System")

# Chatbot interface
st.write("Hi! I'm your recipe recommendation assistant. I'll ask you a few questions to help find the best recipe for you.")

# Initialize session state for conversation and answers
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'answers' not in st.session_state:
    st.session_state.answers = {}

# Function to handle user input
def handle_input(question, key):
    user_input = st.text_input(question, key=key)
    if user_input:
        st.session_state.conversation.append(("You", user_input))
        st.session_state.answers[key] = user_input
        return user_input
    return None

questions = [
    ("So, what taste are you devouring right now?", 'taste'),
    ("What type of food do you generally like?", 'food_type'),
    ("What's your mood currently?", 'mood')
]

# Ask questions one by one
for question, key in questions:
    if key not in st.session_state.answers:
        handle_input(question, key)
        st.stop()

# Display conversation
for speaker, text in st.session_state.conversation:
    if speaker == "You":
        st.text_area("You:", text, height=50, key=speaker+text)

# Function to extract keywords
def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text)
    keywords = [word for word in words if word.isalnum() and word.lower() not in stop_words]
    return keywords

# Extract keywords from all answers
keywords = []
for answer in st.session_state.answers.values():
    keywords.extend(extract_keywords(answer))

# Function to find the best matching recipe
def recommend_recipe(keywords):
    cv = CountVectorizer(max_features=5000)
    vectors = cv.fit_transform(food['tags']).toarray()
    keyword_vector = cv.transform([' '.join(keywords)]).toarray()
    cosine_similarities = cosine_similarity(keyword_vector, vectors).flatten()
    recommended_index = np.argmax(cosine_similarities)
    return recommended_index

# Recommend a recipe
if keywords:
    recommended_index = recommend_recipe(keywords)
    recipe_title = food.iloc[recommended_index]['recipe_title']
    recipe_url = food.iloc[recommended_index]['url']
    st.write(f"I recommend you try: [{recipe_title}]({recipe_url})")
