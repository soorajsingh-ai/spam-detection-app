import pandas as pd


df = pd.read_csv("spam.csv", encoding='latin-1')



#clean dataset
df = df[['v1' , 'v2']]

#rename column
df.columns = ['label' , 'message']


#convert label data to number
df['label'] = df['label'].map({'ham': 0, 'spam':1})

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1,2),
    max_df=0.9,
    min_df=1
)



X = vectorizer.fit_transform(df['message'])
y = df['label']

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression



# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# create model
model = LogisticRegression(max_iter=1000, class_weight='balanced')

# train
model.fit(X_train, y_train)

print(model)

from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

def check_spam(message):
    message =message.lower()
    spam_words = ["win","free","offer","click","prize","money","claim"]
    for word in spam_words:
        if f" {word} " in f" {message} ":
            return "spam 🚫"
        
    vector = vectorizer.transform([message])
    result = model.predict(vector)

    if result[0] == 1:
        return "Spam 🚫"
    else:
        return "Normal ✅"
    
    
    
#print(check_spam("Let's meet tomorrow"))
#print(check_spam("Congratulations! You won a prize"))

import pickle

# save model
pickle.dump(model, open("model.pkl", "wb"))

# save vectorizer
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully ✅")