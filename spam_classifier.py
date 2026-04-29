import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Dataset
data = {
    'message': [
        'Win money now',
        'Call me later',
        'Free gift available',
        'Let’s meet tomorrow',
        'Claim your prize now',
        'Hello friend how are you'
    ],
    'label': ['spam', 'ham', 'spam', 'ham', 'spam', 'ham']
}

df = pd.DataFrame(data)

# Convert text to numbers
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['message'])

y = df['label']

# Train model
model = MultinomialNB()
model.fit(X, y)

# Test message
msg = [input("Enter message : ")]
msg_vector = vectorizer.transform(msg)

prediction = model.predict(msg_vector)

print("Message:", msg[0])
print("Prediction:", prediction[0])

