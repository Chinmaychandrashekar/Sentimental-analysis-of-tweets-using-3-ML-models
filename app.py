import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import nltk
import json
import os
import hashlib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import pipeline

# --- Initial Setup & Configuration ---
st.set_page_config(layout="wide", page_title="Political Tweet Sentiment Analysis", page_icon="📊")

# ---------- AUTHENTICATION HELPERS (LOCAL JSON-BASED) ----------
USERS_FILE = "users.json"

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_users(users_dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users_dict, f, indent=2)

def authenticate_user(username, password):
    users = load_users()
    if username in users:
        stored_hash = users[username]["password"]
        return stored_hash == hash_password(password)
    return False

def create_user(username, password, email=None):
    users = load_users()
    if username in users:
        return False, "Username already exists. Please choose another."
    users[username] = {
        "password": hash_password(password),
        "email": email or ""
    }
    save_users(users)
    return True, "Account created successfully! You can now log in."

# Initialize session state for auth
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

# ---------- NLTK downloads (cached) ----------
@st.cache_resource
def download_nltk_data():
    # Try-find first to avoid repeated downloads
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet')
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')

download_nltk_data()

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ---------- Utility functions ----------
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(lemmatized_tokens)

def plot_sentiment_distribution_streamlit(df, column_name, title=None, cols=1):
    if title is None:
        title = f"Sentiment Distribution: {column_name}"
    fig, ax = plt.subplots(figsize=(6*cols, 4))
    order = ['Positive', 'Negative', 'Neutral']
    sns.countplot(x=column_name, data=df, order=order, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Sentiment", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    st.pyplot(fig)

def plot_confusion_matrix_func(cm, labels, model_name):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    st.pyplot(fig)

@st.cache_resource
def get_bert_pipeline():
    # use distilbert sst2 fine-tuned as a fast default
    return pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')

def get_bert_sentiment(text, _pipeline):
    try:
        result = _pipeline(text)[0]
        if result['label'] == 'POSITIVE' and result['score'] > 0.6:
            return 'Positive'
        elif result['label'] == 'NEGATIVE' and result['score'] > 0.6:
            return 'Negative'
        else:
            return 'Neutral'
    except Exception:
        return 'Neutral'


# ---------- MAIN AUTHENTICATION LAYER ----------
def show_auth_screen():
    st.markdown(
        "<h1 style='text-align:center;'>🔐 Political Tweet Sentiment Portal</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<p style='text-align:center; color:gray;'>Please login or create an account to access the analysis dashboard.</p>",
        unsafe_allow_html=True
    )

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("👤 Login")
        login_username = st.text_input("Username", key="login_username")
        login_password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login", use_container_width=True, type="primary"):
            if not login_username or not login_password:
                st.warning("Please enter both username and password.")
            else:
                if authenticate_user(login_username, login_password):
                    st.session_state.authenticated = True
                    st.session_state.username = login_username
                    st.success(f"Welcome back, {login_username}!")
                    st.rerun()

                else:
                    st.error("Invalid username or password.")

    with col_right:
        st.subheader("🆕 Sign Up")
        signup_username = st.text_input("New Username", key="signup_username")
        signup_email = st.text_input("Email (optional)", key="signup_email")
        signup_password = st.text_input("New Password", type="password", key="signup_password")
        signup_password_confirm = st.text_input("Confirm Password", type="password", key="signup_password_confirm")
        if st.button("Create Account", use_container_width=True):
            if not signup_username or not signup_password or not signup_password_confirm:
                st.warning("Please fill all required fields (username & password).")
            elif signup_password != signup_password_confirm:
                st.error("Passwords do not match.")
            else:
                success, msg = create_user(signup_username, signup_password, signup_email)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)

# ---------- ORIGINAL APP STATE KEYS ----------
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
    st.session_state.vectorizer = None
    st.session_state.models = None
    st.session_state.labels = None

# ---------- RENDER: IF NOT AUTHENTICATED -> LOGIN / SIGNUP ----------
if not st.session_state.authenticated:
    show_auth_screen()
    st.stop()   # Do not run the rest of the app until logged in

# ---------- IF AUTHENTICATED: SHOW ORIGINAL APP ----------
# Top bar / sidebar info for logged-in user
with st.sidebar:
    st.markdown(f"**👋 Logged in as:** `{st.session_state.username}`")
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = None
        st.rerun()


st.title("📊 Sentimental Analysis of Political Tweets")
st.markdown("""
This application analyzes political tweet sentiment using three models:
- Logistic Regression (TF-IDF)
- Multinomial Naive Bayes (TF-IDF)
- Pretrained BERT (pipeline) """)

st.sidebar.header("⚙ Controls")
input_method = st.sidebar.radio("Choose your input method", ["Manual Tweet Input", "Upload CSV File"])
bert_pipeline = get_bert_pipeline()

# ---------- Manual Input ----------
if input_method == "Manual Tweet Input":
    st.header("✍ Manual Tweet Analysis")
    user_tweet = st.text_area("Enter a political tweet below:", height=140)

    if st.button("Analyze Tweet"):
        if not user_tweet:
            st.warning("Please enter a tweet to analyze.")
        else:
            st.subheader("1. Preprocessing Breakdown")
            with st.expander("See the step-by-step text cleaning process", expanded=True):
                cleaned_text_stage1 = re.sub(r'http\S+|www\S+|https\S+', '', user_tweet, flags=re.MULTILINE)
                cleaned_text_stage2 = re.sub(r'\@\w+|\#', '', cleaned_text_stage1)
                cleaned_text_stage3 = re.sub(r'[^A-Za-z\s]', '', cleaned_text_stage2).lower()
                st.markdown(f"**Original:** {user_tweet}")
                st.markdown(f"**Cleaned (URLs, mentions, punctuation removed):** {cleaned_text_stage3}")
                tokens = word_tokenize(cleaned_text_stage3)
                st.markdown(f"**Tokens:** {tokens}")
                tokens_no_stopwords = [word for word in tokens if word not in stop_words]
                st.markdown(f"**Tokens (Stop Words Removed):** {tokens_no_stopwords}")
                lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens_no_stopwords]
                st.markdown(f"**Lemmatized Tokens:** {lemmatized_tokens}")
            final_processed_text = " ".join(lemmatized_tokens)
            st.success(f"Final Text for Models: {final_processed_text}")

            st.subheader("2. Model Predictions")
            col1, col2, col3 = st.columns(3)
            with col1:
                bert_pred = get_bert_sentiment(user_tweet, bert_pipeline)
                st.metric("BERT Prediction", bert_pred)
            if not st.session_state.models_trained:
                st.info("Upload a CSV to train and see predictions from Logistic Regression and Naive Bayes models.")
            else:
                vectorized_tweet = st.session_state.vectorizer.transform([final_processed_text])
                with col2:
                    lr_model = st.session_state.models["Logistic Regression"]
                    lr_pred = lr_model.predict(vectorized_tweet)[0]
                    st.metric("Logistic Regression", lr_pred)
                with col3:
                    nb_model = st.session_state.models["Multinomial Naive Bayes"]
                    nb_pred = nb_model.predict(vectorized_tweet)[0]
                    st.metric("Naive Bayes", nb_pred)

# ---------- CSV Upload & Analysis ----------
elif input_method == "Upload CSV File":
    st.header("📁 Upload and Analyze a Dataset")
    uploaded_file = st.file_uploader("Upload a CSV file containing columns for tweet text and (optional) sentiment label.", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Uploaded file preview")
        st.write(f"Columns detected: {list(df.columns)}")
        st.dataframe(df.head())

        # Attempt to find text and sentiment columns (case-insensitive)
        text_cols = ['tweet', 'tweet_text', 'text', 'body']
        sentiment_cols = ['sentiment', 'label', 'class', 'category']
        lower_cols = [col.lower() for col in df.columns]
        found_cols = {}

        for target in text_cols:
            if target in lower_cols:
                original_name = df.columns[lower_cols.index(target)]
                found_cols['tweet'] = original_name
                break

        for target in sentiment_cols:
            if target in lower_cols:
                original_name = df.columns[lower_cols.index(target)]
                found_cols['sentiment'] = original_name
                break

        # ---------- Labeled dataset branch ----------
        if 'tweet' in found_cols and 'sentiment' in found_cols:
            st.info("Detected labeled dataset. We'll train TF-IDF based models and evaluate them (Logistic Regression & Naive Bayes). BERT is used as a sample evaluator.")
            df = df[[found_cols['tweet'], found_cols['sentiment']]].rename(columns={
                found_cols['tweet']: 'tweet',
                found_cols['sentiment']: 'sentiment'
            }).copy()
            df.dropna(subset=['tweet', 'sentiment'], inplace=True)
            df['sentiment'] = df['sentiment'].astype(str).str.capitalize().str.strip()
            valid_sentiments = ['Positive', 'Negative', 'Neutral']
            df = df[df['sentiment'].isin(valid_sentiments)]
            st.session_state.labels = sorted(df['sentiment'].unique())

            if df.empty:
                st.error("The dataset is empty after filtering for valid sentiments. Please check your data.")
            else:
                st.subheader("1. Data Overview")
                st.write(f"Loaded {len(df)} valid rows from the dataset.")
                st.dataframe(df.head())

                # Show distribution
                with st.expander("Show sentiment distribution", expanded=True):
                    plot_sentiment_distribution_streamlit(df, 'sentiment', title="Original Label Distribution")

                # Preprocessing + TF-IDF
                with st.spinner("Preprocessing data and training models..."):
                    df['processed_tweet'] = df['tweet'].apply(preprocess_text)

                    if len(df) < 5:
                        st.error("Need at least 5 rows to perform a train/test split. Please upload a larger dataset.")
                    else:
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                df['processed_tweet'], df['sentiment'], test_size=0.2,
                                random_state=42, stratify=df['sentiment']
                            )
                        except ValueError:
                            X_train, X_test, y_train, y_test = train_test_split(
                                df['processed_tweet'], df['sentiment'], test_size=0.2, random_state=42
                            )

                        tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
                        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
                        X_test_tfidf = tfidf_vectorizer.transform(X_test)

                        models = {
                            "Logistic Regression": LogisticRegression(max_iter=1000, solver='liblinear'),
                            "Multinomial Naive Bayes": MultinomialNB()
                        }
                        st.session_state.vectorizer = tfidf_vectorizer
                        st.session_state.models = models
                        st.session_state.models_trained = True

                        performance_metrics = {}

                        st.subheader("2. Model Performance Analysis")
                        tab1, tab2, tab3 = st.tabs(["Logistic Regression", "Multinomial Naive Bayes", "BERT Model"])

                        for model_name, model in models.items():
                            model.fit(X_train_tfidf, y_train)
                            y_pred = model.predict(X_test_tfidf)

                            report = classification_report(y_test, y_pred, output_dict=True, labels=st.session_state.labels, zero_division=0)
                            performance_metrics[model_name] = report

                            if model_name == "Logistic Regression":
                                with tab1:
                                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                                    st.text("Classification Report:")
                                    st.dataframe(pd.DataFrame(report).transpose())
                                    cm = confusion_matrix(y_test, y_pred, labels=st.session_state.labels)
                                    plot_confusion_matrix_func(cm, st.session_state.labels, model_name)
                                    # Per-class bar chart: precision/recall/f1
                                    prf = pd.DataFrame(report).transpose().loc[st.session_state.labels]
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    prf[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
                                    ax.set_title("Per-class Precision / Recall / F1 - Logistic Regression")
                                    ax.set_ylim(0, 1)
                                    st.pyplot(fig)

                            else:
                                with tab2:
                                    st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.2%}")
                                    st.text("Classification Report:")
                                    st.dataframe(pd.DataFrame(report).transpose())
                                    cm = confusion_matrix(y_test, y_pred, labels=st.session_state.labels)
                                    plot_confusion_matrix_func(cm, st.session_state.labels, model_name)
                                    prf = pd.DataFrame(report).transpose().loc[st.session_state.labels]
                                    fig, ax = plt.subplots(figsize=(8, 4))
                                    prf[['precision', 'recall', 'f1-score']].plot(kind='bar', ax=ax)
                                    ax.set_title("Per-class Precision / Recall / F1 - MultinomialNB")
                                    ax.set_ylim(0, 1)
                                    st.pyplot(fig)

                        # BERT (sampled)
                        with tab3:
                            st.info("BERT analysis is run on a sample of the test set (max 200) for speed.")
                            sample_size = min(200, len(X_test))
                            if sample_size > 0:
                                sample_X_test = X_test.sample(n=sample_size, random_state=1)
                                sample_y_test = y_test[sample_X_test.index]
                                bert_preds = [get_bert_sentiment(text, bert_pipeline) for text in sample_X_test]
                                report = classification_report(sample_y_test, bert_preds, output_dict=True, labels=st.session_state.labels, zero_division=0)
                                performance_metrics["BERT"] = report
                                st.metric("Accuracy (on sample)", f"{accuracy_score(sample_y_test, bert_preds):.2%}")
                                st.text("Classification Report (on sample):")
                                st.dataframe(pd.DataFrame(report).transpose())
                                cm = confusion_matrix(sample_y_test, bert_preds, labels=st.session_state.labels)
                                plot_confusion_matrix_func(cm, st.session_state.labels, "BERT (sample)")
                            else:
                                st.warning("The test set is too small to run the BERT analysis sample.")

                        st.success("Models trained successfully!")

                        # Comparison of F1 scores
                        st.subheader("3. Models Performance Comparison (F1 - Macro Avg)")
                        f1_scores = {}
                        for model, metrics in performance_metrics.items():
                            # handle nested dicts for scikit report vs BERT sample
                            try:
                                f1 = metrics['macro avg']['f1-score']
                            except Exception:
                                f1 = np.nan
                            f1_scores[model] = f1
                        f1_df = pd.DataFrame.from_dict(f1_scores, orient='index', columns=['F1-Score (Macro Avg)'])
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x=f1_df.index, y=f1_df['F1-Score (Macro Avg)'], ax=ax)
                        ax.set_title('Comparison of F1-Scores Across Models')
                        ax.set_ylim(0, 1)
                        st.pyplot(fig)

                        # Inter-model agreement (on test set) + majority vote vs true
                        st.subheader("4. Inter-model Agreement & Consensus (on test set)")
                        # Predict full test set using trained models
                        lr_preds_full = models["Logistic Regression"].predict(X_test_tfidf)
                        nb_preds_full = models["Multinomial Naive Bayes"].predict(X_test_tfidf)
                        bert_preds_full = [get_bert_sentiment(text, bert_pipeline) for text in X_test]

                        agreement_df = pd.DataFrame({
                            'true': y_test.values,
                            'lr': lr_preds_full,
                            'nb': nb_preds_full,
                            'bert': bert_preds_full
                        }, index=y_test.index)

                        def majority_vote_row(r):
                            preds = [r['lr'], r['nb'], r['bert']]
                            return max(set(preds), key=preds.count)

                        agreement_df['majority'] = agreement_df.apply(majority_vote_row, axis=1)
                        # Agreement ratios
                        agree_lr_nb = (agreement_df['lr'] == agreement_df['nb']).mean() * 100
                        agree_lr_bert = (agreement_df['lr'] == agreement_df['bert']).mean() * 100
                        agree_nb_bert = (agreement_df['nb'] == agreement_df['bert']).mean() * 100
                        st.write(f"LR vs NB agreement: {agree_lr_nb:.2f}%")
                        st.write(f"LR vs BERT agreement: {agree_lr_bert:.2f}%")
                        st.write(f"NB vs BERT agreement: {agree_nb_bert:.2f}%")

                        # Majority vs true metrics
                        mv_precision, mv_recall, mv_f1, _ = precision_recall_fscore_support(
                            agreement_df['true'], agreement_df['majority'], labels=st.session_state.labels, zero_division=0, average='macro'
                        )
                        st.write("Majority vote - Macro precision/recall/f1:")
                        st.write({
                            'precision_macro': round(mv_precision, 3),
                            'recall_macro': round(mv_recall, 3),
                            'f1_macro': round(mv_f1, 3)
                        })

                        st.success("Labeled dataset evaluation complete. Download trained models' vectorizer state and predictions as needed.")

                        # Save predictions on test set for download
                        preds_out = agreement_df.copy()
                        preds_out['index'] = preds_out.index
                        preds_csv = preds_out.to_csv(index=False)
                        st.download_button("Download test-set predictions & consensus as CSV", data=preds_csv, file_name="labeled_test_predictions_consensus.csv")

        # ---------- Unlabeled dataset branch ----------
        elif 'tweet' in found_cols:
            st.info("Detected unlabeled dataset. We'll predict using available models (if trained) and BERT. You will get distribution graphs and agreement analysis.")
            df = df[[found_cols['tweet']]].rename(columns={found_cols['tweet']: 'tweet'}).copy()
            df.dropna(subset=['tweet'], inplace=True)
            st.subheader("1. Data Overview")
            st.write(f"Loaded {len(df)} rows from the dataset.")
            st.dataframe(df.head())

            with st.spinner("Preprocessing tweets..."):
                df['processed_tweet'] = df['tweet'].apply(preprocess_text)

            # Predict using saved TF-IDF models if available
            if st.session_state.models_trained and st.session_state.vectorizer is not None:
                vec = st.session_state.vectorizer
                X_tfidf = vec.transform(df['processed_tweet'])
                lr_model = st.session_state.models["Logistic Regression"]
                nb_model = st.session_state.models["Multinomial Naive Bayes"]
                st.info("Using trained Logistic Regression and Naive Bayes models from session.")
                df['lr_pred'] = lr_model.predict(X_tfidf)
                df['nb_pred'] = nb_model.predict(X_tfidf)
            else:
                st.warning("No trained TF-IDF models found in session. LR/NB predictions will be marked 'Not trained'.")
                df['lr_pred'] = "Not trained"
                df['nb_pred'] = "Not trained"

            # BERT predictions (always available)
            with st.spinner("Running BERT predictions (this may take a while for large datasets)..."):
                df['bert_pred'] = df['tweet'].apply(lambda x: get_bert_sentiment(x, bert_pipeline))

            st.subheader("2. Prediction Results Overview")
            # show counts
            models_for_counts = ['lr_pred', 'nb_pred', 'bert_pred']
            counts = {}
            for col in models_for_counts:
                if col in df.columns:
                    counts[col] = df[col].value_counts().to_dict()
            st.write("Predicted sentiment counts by model:")
            st.json(counts)

            # Visualization: distribution per model
            st.subheader("3. Sentiment Distribution Visualization (Predictions)")
            cols_available = [c for c in ['lr_pred', 'nb_pred', 'bert_pred'] if c in df.columns and not (df[c] == "Not trained").all()]
            if len(cols_available) == 0:
                st.warning("No model predictions available for visualization.")
            else:
                # Create subplots side-by-side (cap at 3)
                n_cols = len(cols_available)
                fig, axes = plt.subplots(1, n_cols, figsize=(6*n_cols, 4))
                if n_cols == 1:
                    axes = [axes]
                order = ['Positive', 'Negative', 'Neutral']
                for i, col in enumerate(cols_available):
                    sns.countplot(x=col, data=df, order=order, ax=axes[i])
                    axes[i].set_title(f"{col} Distribution")
                    axes[i].set_xlabel("Sentiment")
                    axes[i].set_ylabel("Count")
                st.pyplot(fig)

            # Inter-model agreement analysis (unlabeled)
            st.subheader("4. Inter-model Agreement Analysis (Unlabeled)")
            # majority vote (works even if LR/NB show "Not trained" — they'll be counted as different tokens)
            def majority_vote(row):
                preds = [row.get('lr_pred'), row.get('nb_pred'), row.get('bert_pred')]
                preds = [p for p in preds if p is not None]
                if len(preds) == 0:
                    return "No prediction"
                try:
                    return max(set(preds), key=preds.count)
                except Exception:
                    return preds[0]

            df['majority_vote'] = df.apply(majority_vote, axis=1)
            majority_counts = df['majority_vote'].value_counts()
            st.write("Majority-vote distribution (consensus across available models):")
            st.dataframe(majority_counts)

            # Agreement crosstab (LR vs NB vs BERT)
            st.write("Model agreement crosstab (LR vs NB vs BERT):")
            try:
                agreement_matrix = pd.crosstab(df['lr_pred'], [df['nb_pred'], df['bert_pred']])
                st.dataframe(agreement_matrix)
            except Exception:
                st.write("Not enough diverse model columns available to build a full crosstab.")

            # Consistency percentages
            st.subheader("5. Consistency / Alignment Scores")
            df['agree_lr_nb'] = df['lr_pred'] == df['nb_pred']
            df['agree_lr_bert'] = df['lr_pred'] == df['bert_pred']
            df['agree_nb_bert'] = df['nb_pred'] == df['bert_pred']

            lr_nb = df.loc[(df['lr_pred'] != "Not trained") & (df['nb_pred'] != "Not trained"), 'lr_pred'] == df.loc[(df['lr_pred'] != "Not trained") & (df['nb_pred'] != "Not trained"), 'nb_pred']
            lr_bert = df.loc[(df['lr_pred'] != "Not trained") & (df['bert_pred'].notna()), 'lr_pred'] == df.loc[(df['lr_pred'] != "Not trained") & (df['bert_pred'].notna()), 'bert_pred']
            nb_bert = df.loc[(df['nb_pred'] != "Not trained") & (df['bert_pred'].notna()), 'nb_pred'] == df.loc[(df['nb_pred'] != "Not trained") & (df['bert_pred'].notna()), 'bert_pred']

            scores = {
                'LR vs NB agreement (%)': round(lr_nb.mean()*100, 2) if lr_nb.size > 0 else np.nan,
                'LR vs BERT agreement (%)': round(lr_bert.mean()*100, 2) if lr_bert.size > 0 else np.nan,
                'NB vs BERT agreement (%)': round(nb_bert.mean()*100, 2) if nb_bert.size > 0 else np.nan
            }
            st.dataframe(pd.DataFrame.from_dict(scores, orient='index', columns=['Agreement %']))

            # Show sample of consensus rows
            with st.expander("Show sample rows with model predictions and majority vote", expanded=False):
                st.dataframe(df[['tweet', 'lr_pred', 'nb_pred', 'bert_pred', 'majority_vote']].sample(n=min(10, len(df)), random_state=1))

            # Download predictions
            st.info("Saving prediction results; use download below for further analysis.")
            csv = df.to_csv(index=False)
            st.download_button("Download predictions as CSV", data=csv, file_name="tweet_sentiments_predicted.csv")

            st.success("Analysis completed for unlabeled data. Note: precision/recall/F1 require true labels; here we provide distributions and agreement metrics as proxies.")

        else:
            st.error("Error: Could not find required columns. Please ensure your CSV contains a text column (e.g., 'tweet', 'text').")
