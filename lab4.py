import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# =========================
# TEXT PREPROCESSING
# =========================
stop_words = set([
    "và","là","của","có","cho","với","rất","một","những","các","được",
    "to","the","and","is","in","it","of","for","on"
])

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-ZÀ-ỹ\s]', '', text)  # bỏ ký tự đặc biệt
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return tokens


# =========================
# HÀM CHUNG
# =========================
def encode_categorical(df, cols):
    df_encoded = df.copy()
    le = LabelEncoder()
    for col in cols:
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def tfidf_transform(text_data):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)
    return X, vectorizer

def train_word2vec(tokenized_text):
    model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1)
    return model


# =========================
# BÀI 1: HOTEL REVIEW
# =========================
print("===== BÀI 1 =====")

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_4_Hotel_reviews.csv")

print("Missing values:\n", df.isnull().sum())

tokens1 = df['review_text'].apply(preprocess_text)

# Fix NaN cho text
df['review_text'] = df['review_text'].fillna('')

cols_cat = ['hotel_name', 'customer_type']

le = LabelEncoder()
for col in cols_cat:
    df[col] = le.fit_transform(df[col].astype(str))

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['review_text'])

print("TF-IDF shape:", tfidf_matrix.shape)

w2v_model = Word2Vec(
    sentences=tokens1,
    vector_size=100,
    window=5,
    min_count=1
)

print("\nTừ gần nghĩa với 'sạch':")

if "sạch" in w2v_model.wv:
    print(w2v_model.wv.most_similar("sạch", topn=5))
else:
    print("Không có từ 'sạch' trong vocabulary")


# =========================
# BÀI 2: MATCH COMMENT
# =========================
print("===== BÀI 2 =====")

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_4_Match_comments.csv")

df['comment_text'] = df['comment_text'].fillna('')

cols_cat = ['team','author']
df2 = encode_categorical(df, cols_cat)

tokens2 = df['comment_text'].apply(preprocess_text)

tfidf2, vec2 = tfidf_transform(df['comment_text'])
print("Shape:", tfidf2.shape)

w2v2 = train_word2vec(tokens2)

print("Gần nghĩa 'xuất sắc':")
if "xuất" in w2v2.wv:
    print(w2v2.wv.most_similar("xuất", topn=5))


# =========================
# BÀI 3: PLAYER FEEDBACK
# =========================
print("===== BÀI 3 =====")

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_4_Player_feedback.csv")

df['feedback_text'] = df['feedback_text'].fillna('')

cols_cat = ['player_type','device']
df3 = encode_categorical(df, cols_cat)

tokens3 = df['feedback_text'].apply(preprocess_text)

tfidf3, vec3 = tfidf_transform(df['feedback_text'])
print("Shape:", tfidf3.shape)

w2v3 = train_word2vec(tokens3)

print("Gần nghĩa 'đẹp':")
if "đẹp" in w2v3.wv:
    print(w2v3.wv.most_similar("đẹp", topn=5))


# =========================
# BÀI 4: ALBUM REVIEW
# =========================
print("===== BÀI 4 =====")

df = pd.read_csv(r"C:\Users\Ngopt\Downloads\ITA105_Lab\ITA105_Lab_4_Album_reviews.csv")

df['review_text'] = df['review_text'].fillna('')

cols_cat = ['genre','platform']
df4 = encode_categorical(df, cols_cat)

tokens4 = df['review_text'].apply(preprocess_text)

tfidf4, vec4 = tfidf_transform(df['review_text'])
print("Shape:", tfidf4.shape)

w2v4 = train_word2vec(tokens4)

print("Gần nghĩa 'sáng tạo':")
if "sáng" in w2v4.wv:
    print(w2v4.wv.most_similar("sáng", topn=5))