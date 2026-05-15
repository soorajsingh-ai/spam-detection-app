import streamlit as st
import pickle

# load saved files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# UI
st.title("📩 Spam Detection App")

msg = st.text_input("Enter your message")

if st.button("Check"):
    if msg:
        msg_vector = vectorizer.transform([msg])
        prediction = model.predict(msg_vector)

        if prediction[0] == 1:
            st.error("Spam Message 🚫")
        else:
            st.success("Normal Message ✅")