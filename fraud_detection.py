import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import *
from sentence_transformers import SentenceTransformer
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🚨 Fraud Detection using ML & BERT")

@st.cache_data
def generate_data(seed=42, n_samples=60000):
    np.random.seed(seed)
    transactions = np.random.choice([
        "purchase at grocery", "online subscription", "electronics store",
        "withdrawal at atm", "transfer to account", "suspicious login", "bitcoin purchase",
        "gift card purchase", "frequent small charges"
    ], size=n_samples)

    labels = np.random.choice([0, 1], size=n_samples, p=[0.97, 0.03])

    data = pd.DataFrame({
        "amount": np.random.exponential(scale=100, size=n_samples),
        "user_age": np.random.randint(18, 80, size=n_samples),
        "user_account_age_days": np.random.randint(30, 2000, size=n_samples),
        "transaction": transactions,
        "isfraud": labels
    })

    return data

# Load data
data = generate_data()

# Display data
st.subheader("📊 Sample Data")
st.dataframe(data.head())  # Show the first 5 rows

X = data.drop(columns=['isfraud'])
y = data['isfraud']

# TF-IDF + XGBoost
tfidf = TfidfVectorizer(max_features=100)
x_text = tfidf.fit_transform(X['transaction'])

x_numeric = X.drop(columns='transaction').values
x_full = np.hstack((x_numeric, x_text.toarray()))

xtrain, xtest, ytrain, ytest = train_test_split(x_full, y, test_size=0.3, random_state=42)

xgb_model = XGBClassifier(
    scale_pos_weight=(ytrain == 0).sum() / (ytrain == 1).sum(),
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)
xgb_model.fit(xtrain, ytrain)
ypred = xgb_model.predict(xtest)
yprobs = xgb_model.predict_proba(xtest)[:, 1]

f1 = f1_score(ytest, ypred)
conf = confusion_matrix(ytest, ypred)
report = classification_report(ytest, ypred)

# BERT + RandomForest
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
x_bert_text = bert_model.encode(X['transaction'].tolist(), show_progress_bar=False)
x_bert_full = np.hstack((x_numeric, x_bert_text))

X_train_bert, X_test_bert, y_train_bert, y_test_bert = train_test_split(
    x_bert_full, y, test_size=0.2, stratify=y, random_state=42
)

rf_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_model.fit(X_train_bert, y_train_bert)
y_pred_bert = rf_model.predict(X_test_bert)

f1_bert = f1_score(y_test_bert, y_pred_bert)
conf_matrix_bert = confusion_matrix(y_test_bert, y_pred_bert)
report_bert = classification_report(y_test_bert, y_pred_bert)

# SMOTE + TF-IDF + Random Forest
xtrain_tfidf, xtest_tfidf, ytrain_tfidf, ytest_tfidf = train_test_split(
    x_full, y, test_size=0.3, random_state=42, stratify=y
)

sm = SMOTE(random_state=42)
xtrain_sm, ytrain_sm = sm.fit_resample(xtrain_tfidf, ytrain_tfidf)

rf_sm_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_sm_model.fit(xtrain_sm, ytrain_sm)

ypred_sm = rf_sm_model.predict(xtest_tfidf)
f1_sm = f1_score(ytest_tfidf, ypred_sm)
conf_sm = confusion_matrix(ytest_tfidf, ypred_sm)
report_sm = classification_report(ytest_tfidf, ypred_sm)

# Show Results
st.subheader("🔍 Model Comparison")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### TF-IDF + XGBoost")
    st.write("F1 Score:", round(f1, 4))
    st.text("Confusion Matrix")
    st.write(conf)
    st.text("Classification Report")
    st.text(report)

with col2:
    st.markdown("### BERT + Random Forest")
    st.write("F1 Score:", round(f1_bert, 4))
    st.text("Confusion Matrix")
    st.write(conf_matrix_bert)
    st.text("Classification Report")
    st.text(report_bert)

with col3:
    st.markdown("### SMOTE + TF-IDF + Random Forest")
    st.write("F1 Score:", round(f1_sm, 4))
    st.text("Confusion Matrix")
    st.write(conf_sm)
    st.text("Classification Report")
    st.text(report_sm)

# Conclusion: Which model is better?
st.subheader("📝 Conclusion")

if f1 > f1_bert and f1 > f1_sm:
    st.write("### Best Model: **TF-IDF + XGBoost**")
    st.write("**Why:** The TF-IDF + XGBoost model shows the highest F1 score among all models, suggesting that it provides a good balance between precision and recall for fraud detection.")
elif f1_bert > f1 and f1_bert > f1_sm:
    st.write("### Best Model: **BERT + Random Forest**")
    st.write("**Why:** Although the F1 score is slightly lower, the BERT embeddings can capture deeper semantic features in transaction descriptions, which may be helpful in understanding more complex patterns.")
else:
    st.write("### Best Model: **SMOTE + TF-IDF + Random Forest**")
    st.write("**Why:** SMOTE helps to balance the class distribution, improving the performance of Random Forest by dealing with class imbalance, leading to a slightly better overall performance.")

# Precision-Recall Curve for XGBoost
precision, recall, thresholds = precision_recall_curve(ytest, yprobs)

fig, ax = plt.subplots()
ax.plot(thresholds, precision[:-1], "b--", label="Precision")
ax.plot(thresholds, recall[:-1], "g-", label="Recall")
ax.set_xlabel("Threshold")
ax.set_ylabel("Score")
ax.set_title("Precision-Recall vs Threshold")
ax.legend()
ax.grid()

st.subheader("📈 Precision-Recall Curve (TF-IDF + XGBoost)")
st.pyplot(fig)
