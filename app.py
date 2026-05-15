model_path = os.path.join(BASE_DIR, "model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "vectorizer.pkl")

if not os.path.exists(model_path):
    st.error("model.pkl file not found ❌")
if not os.path.exists(vectorizer_path):
    st.error("vectorizer.pkl file not found ❌")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))
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