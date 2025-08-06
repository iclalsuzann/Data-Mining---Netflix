import os
import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# --- Config --- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Plot configurations
plt.style.use('default')
sns.set_palette("husl")

# --- NLTK Setup --- #
nltk.download('stopwords')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
base_stopwords = set(stopwords.words('english'))

# --- Load Data --- #
df = pd.read_csv("netflix_titles.csv")
print(f"Original dataset shape: {df.shape}")

# Data cleaning with visualization
print("\n=== Data Quality Analysis ===")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Missing data visualization
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_data = missing_data[missing_data > 0]
if len(missing_data) > 0:
    axes[0,0].bar(range(len(missing_data)), missing_data.values)
    axes[0,0].set_xticks(range(len(missing_data)))
    axes[0,0].set_xticklabels(missing_data.index, rotation=45)
    axes[0,0].set_title('Missing Values by Column')
    axes[0,0].set_ylabel('Count')
else:
    axes[0,0].text(0.5, 0.5, 'No missing values', ha='center', va='center', transform=axes[0,0].transAxes)
    axes[0,0].set_title('Missing Values by Column')

# Content type distribution
if 'type' in df.columns:
    type_counts = df['type'].value_counts()
    axes[0,1].pie(type_counts.values, labels=type_counts.index, autopct='%1.1f%%')
    axes[0,1].set_title('Content Type Distribution')
else:
    axes[0,1].text(0.5, 0.5, 'Type column not found', ha='center', va='center', transform=axes[0,1].transAxes)
    axes[0,1].set_title('Content Type Distribution')

# Release year distribution
if 'release_year' in df.columns:
    df_temp = df.dropna(subset=['release_year'])
    axes[1,0].hist(df_temp['release_year'], bins=30, alpha=0.7, edgecolor='black')
    axes[1,0].set_title('Release Year Distribution')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Count')
else:
    axes[1,0].text(0.5, 0.5, 'Release year not found', ha='center', va='center', transform=axes[1,0].transAxes)
    axes[1,0].set_title('Release Year Distribution')

# Description length distribution (before cleaning)
desc_lengths = df.dropna(subset=['description'])['description'].str.len()
axes[1,1].hist(desc_lengths, bins=50, alpha=0.7, edgecolor='black')
axes[1,1].set_title('Description Length Distribution')
axes[1,1].set_xlabel('Characters')
axes[1,1].set_ylabel('Count')

plt.tight_layout()
plt.savefig("data_quality_analysis.png", dpi=300, bbox_inches='tight')

# Clean the data
df = df.dropna(subset=["description", "listed_in"])
df["description"] = df["description"].str.strip()
df = df[df["description"].str.len() > 10]
df = df.drop_duplicates(subset=["description"])
print(f"After cleaning: {df.shape}")

# --- Genre Mapping with more specific labels --- #
def refined_genre_map(x):
    x = x.lower()
    if "comedy" in x: return "Comedy"
    elif "drama" in x: return "Drama"
    elif "action" in x: return "Action"
    elif "documentary" in x or "docuseries" in x: return "Documentary"
    elif "kids" in x or "children" in x: return "Kids"
    elif "reality" in x: return "Reality"
    elif "horror" in x: return "Horror"
    elif "anime" in x: return "Anime"
    elif "romance" in x: return "Romance"
    elif "thriller" in x: return "Thriller"
    else: return "Other"

df["primary_genre"] = df["listed_in"].apply(refined_genre_map)

# Genre distribution visualization BEFORE filtering
plt.figure(figsize=(12, 6))
genre_counts_all = df["primary_genre"].value_counts()
plt.subplot(1, 2, 1)
plt.bar(range(len(genre_counts_all)), genre_counts_all.values)
plt.xticks(range(len(genre_counts_all)), genre_counts_all.index, rotation=45)
plt.title('Genre Distribution (All)')
plt.ylabel('Count')

# Remove classes with fewer than 30 samples
genre_counts = df["primary_genre"].value_counts()
valid_genres = genre_counts[genre_counts >= 30].index
df = df[df["primary_genre"].isin(valid_genres)].reset_index(drop=True)

# Genre distribution visualization AFTER filtering
genre_counts_filtered = df["primary_genre"].value_counts()
plt.subplot(1, 2, 2)
colors = plt.cm.Set3(np.linspace(0, 1, len(genre_counts_filtered)))
bars = plt.bar(range(len(genre_counts_filtered)), genre_counts_filtered.values, color=colors)
plt.xticks(range(len(genre_counts_filtered)), genre_counts_filtered.index, rotation=45)
plt.title('Genre Distribution (Filtered ≥30 samples)')
plt.ylabel('Count')

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig("genre_distribution.png", dpi=300, bbox_inches='tight')

print(f"Final dataset shape: {df.shape}")
print(f"Number of genres: {len(valid_genres)}")

# --- Text Preprocessing --- #

# Netflix specific stopwords added to base
netflix_stopwords = base_stopwords.union({
    'season', 'episode', 'series', 'netflix', 'show', 'character', 'characters',
    'story', 'based', 'new', 'world', 'life', 'love', 'time', 'year', 'one', 'two',
    'man', 'woman', 'people', 'family', 'friend'
})

def advanced_clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in netflix_stopwords]
    return " ".join(tokens)

df["clean_desc"] = df["description"].apply(advanced_clean_text)

# Text analysis visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Description length comparison (before vs after cleaning)
original_lengths = df['description'].str.split().apply(len)
clean_lengths = df['clean_desc'].str.split().apply(len)

axes[0,0].hist(original_lengths, alpha=0.5, label='Original', bins=30)
axes[0,0].hist(clean_lengths, alpha=0.5, label='Cleaned', bins=30)
axes[0,0].set_title('Description Length: Before vs After Cleaning')
axes[0,0].set_xlabel('Number of Words')
axes[0,0].set_ylabel('Frequency')
axes[0,0].legend()

# Average description length by genre
avg_lengths = df.groupby('primary_genre')['clean_desc'].apply(lambda x: x.str.split().apply(len).mean())
axes[0,1].bar(range(len(avg_lengths)), avg_lengths.values)
axes[0,1].set_xticks(range(len(avg_lengths)))
axes[0,1].set_xticklabels(avg_lengths.index, rotation=45)
axes[0,1].set_title('Average Description Length by Genre')
axes[0,1].set_ylabel('Average Words')

# Most common words across all descriptions
from collections import Counter
all_words = ' '.join(df['clean_desc']).split()
word_freq = Counter(all_words)
top_words = dict(word_freq.most_common(15))

axes[1,0].bar(range(len(top_words)), list(top_words.values()))
axes[1,0].set_xticks(range(len(top_words)))
axes[1,0].set_xticklabels(list(top_words.keys()), rotation=45)
axes[1,0].set_title('Top 15 Most Common Words')
axes[1,0].set_ylabel('Frequency')

# Word count distribution by genre (boxplot)
df_melted = []
for genre in df['primary_genre'].unique():
    genre_data = df[df['primary_genre'] == genre]['clean_desc'].str.split().apply(len)
    df_melted.extend([(genre, length) for length in genre_data])

df_plot = pd.DataFrame(df_melted, columns=['Genre', 'Word_Count'])
sns.boxplot(data=df_plot, x='Genre', y='Word_Count', ax=axes[1,1])
axes[1,1].set_title('Word Count Distribution by Genre')
axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)

plt.tight_layout()
plt.savefig("text_analysis.png", dpi=300, bbox_inches='tight')

# --- Feature Engineering --- #
df['desc_len'] = df['clean_desc'].str.split().apply(len)

# Example keywords for additional features
keywords = ['love', 'fight', 'school', 'family', 'murder', 'friend', 'war', 'secret']
for kw in keywords:
    df[f'kw_{kw}'] = df['clean_desc'].apply(lambda x: x.count(kw))

# Keyword analysis visualization
keyword_data = df[[f'kw_{kw}' for kw in keywords] + ['primary_genre']]
keyword_means = keyword_data.groupby('primary_genre').mean()

plt.figure(figsize=(12, 8))
sns.heatmap(keyword_means.T, annot=True, cmap='YlOrRd', fmt='.2f')
plt.title('Average Keyword Frequency by Genre')
plt.xlabel('Genre')
plt.ylabel('Keywords')
plt.tight_layout()
plt.savefig("keyword_heatmap.png", dpi=300, bbox_inches='tight')

# Release year - fill missing with median or zero if not present
if 'release_year' in df.columns:
    df['release_year'] = df['release_year'].fillna(df['release_year'].median())
    
    # Release year vs Genre visualization
    plt.figure(figsize=(12, 6))
    for i, genre in enumerate(df['primary_genre'].unique()):
        genre_data = df[df['primary_genre'] == genre]['release_year'].dropna()
        plt.scatter([i] * len(genre_data), genre_data, alpha=0.6, s=20)
    
    plt.xticks(range(len(df['primary_genre'].unique())), df['primary_genre'].unique(), rotation=45)
    plt.ylabel('Release Year')
    plt.title('Release Year Distribution by Genre')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("release_year_by_genre.png", dpi=300, bbox_inches='tight')
else:
    df['release_year'] = 0

# --- Label Encoding --- #
le = LabelEncoder()
df["label"] = le.fit_transform(df["primary_genre"])

# --- Vectorization --- #
vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1,2))
X_tfidf = vectorizer.fit_transform(df["clean_desc"])

# Combine TF-IDF features with extra numeric features
extra_features = df[['desc_len'] + [f'kw_{kw}' for kw in keywords] + ['release_year']].values
X_full = hstack([X_tfidf, extra_features])

y = df["label"].values

# Numeric features distribution by genre
plt.figure(figsize=(15, 8))
numeric_features = ['desc_len'] + [f'kw_{kw}' for kw in keywords]
n_features = len(numeric_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

for i, feature in enumerate(numeric_features):
    plt.subplot(n_rows, n_cols, i + 1)
    for genre in df['primary_genre'].unique():
        genre_data = df[df['primary_genre'] == genre][feature]
        plt.hist(genre_data, alpha=0.6, label=genre, bins=10)
    plt.title(f'{feature} Distribution')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    if i == 0:  # Only show legend for first subplot
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig("numeric_features_distribution.png", dpi=300, bbox_inches='tight')

# --- Handle Class Imbalance with SMOTE --- #
X_dense = X_full.toarray()  # np.array tipi SMOTE ile uyumlu
sm = SMOTE(random_state=SEED)
X_res, y_res = sm.fit_resample(X_dense, y)

print(f"Before SMOTE: {X_full.shape}, {y.shape}")
print(f"After SMOTE: {X_res.shape}, {y_res.shape}")

# Class balance visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Before SMOTE
unique_original, counts_original = np.unique(y, return_counts=True)
genre_names = [le.classes_[i] for i in unique_original]
ax1.bar(genre_names, counts_original)
ax1.set_title('Class Distribution - Before SMOTE')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# After SMOTE
unique_resampled, counts_resampled = np.unique(y_res, return_counts=True)
genre_names_res = [le.classes_[i] for i in unique_resampled]
ax2.bar(genre_names_res, counts_resampled, color='orange')
ax2.set_title('Class Distribution - After SMOTE')
ax2.set_ylabel('Count')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig("class_balance_smote.png", dpi=300, bbox_inches='tight')

# --- Train-Test Split --- #
X_train, X_val, y_train, y_val = train_test_split(X_res, y_res, stratify=y_res, random_state=SEED, test_size=0.2)

# --- Hyperparameter Optimization for Logistic Regression --- #
param_grid = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'lbfgs']
}

lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=SEED)
grid = GridSearchCV(lr, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

print("Best Logistic Regression Params:", grid.best_params_)

# Hyperparameter tuning results visualization
results_df = pd.DataFrame(grid.cv_results_)
plt.figure(figsize=(10, 6))

# Group by C parameter and plot mean scores
c_values = results_df['param_C'].unique()
solvers = results_df['param_solver'].unique()

for solver in solvers:
    solver_data = results_df[results_df['param_solver'] == solver]
    plt.plot(solver_data['param_C'], solver_data['mean_test_score'], 
             marker='o', label=f'solver: {solver}')

plt.xlabel('C Parameter')
plt.ylabel('Mean CV Accuracy')
plt.title('Hyperparameter Tuning Results')
plt.xscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("hyperparameter_tuning.png", dpi=300, bbox_inches='tight')

# --- Define Other Models --- #
nb = MultinomialNB()
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=SEED)

# --- Ensemble Voting Classifier --- #
ensemble = VotingClassifier(
    estimators=[
        ('lr', grid.best_estimator_),
        ('nb', nb),
        ('rf', rf)
    ],
    voting='soft',
    n_jobs=-1
)

ensemble.fit(X_train, y_train)

# --- Individual Model Performance Comparison --- #
models = {
    'Logistic Regression': grid.best_estimator_,
    'Naive Bayes': nb,
    'Random Forest': rf,
    'Ensemble': ensemble
}

model_scores = {}
for name, model in models.items():
    if name != 'Ensemble':  # Ensemble is already fitted
        model.fit(X_train, y_train)
    
    y_pred_temp = model.predict(X_val)
    score = accuracy_score(y_val, y_pred_temp)
    model_scores[name] = score

# Model comparison visualization
plt.figure(figsize=(10, 6))
model_names = list(model_scores.keys())
scores = list(model_scores.values())
colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold']

bars = plt.bar(model_names, scores, color=colors)
plt.title('Model Performance Comparison')
plt.ylabel('Validation Accuracy')
plt.ylim(0, 1)

# Add value labels on bars
for bar, score in zip(bars, scores):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{score:.3f}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300, bbox_inches='tight')

# --- Evaluation --- #
y_pred = ensemble.predict(X_val)
print("\n=== Ensemble Model Classification Report ===")
print(classification_report(y_val, y_pred, target_names=le.classes_))

# --- Enhanced Confusion Matrix --- #
plt.figure(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_val, y_pred, display_labels=le.classes_, xticks_rotation='vertical', 
    cmap='Blues', values_format='d')
plt.title("Ensemble Model - Confusion Matrix", fontsize=14)
plt.tight_layout()
plt.savefig("confusion_matrix_ensemble.png", dpi=300, bbox_inches='tight')

# --- Classification Report as Heatmap --- #
from sklearn.metrics import classification_report
report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).iloc[:-1, :-1].T  # Remove 'accuracy' row and 'support' column

plt.figure(figsize=(10, 8))
sns.heatmap(report_df, annot=True, cmap='RdYlGn', fmt='.3f', center=0.5)
plt.title('Classification Report Heatmap')
plt.tight_layout()
plt.savefig("classification_report_heatmap.png", dpi=300, bbox_inches='tight')

# --- Save classes for later use --- #
np.save("genre_classes.npy", le.classes_)

print("\nPipeline complete. Generated visualization files:")
print(" - data_quality_analysis.png")
print(" - genre_distribution.png") 
print(" - text_analysis.png")
print(" - keyword_heatmap.png")
print(" - release_year_by_genre.png")
print(" - numeric_features_distribution.png")
print(" - class_balance_smote.png")
print(" - hyperparameter_tuning.png")
print(" - model_comparison.png")
print(" - confusion_matrix_ensemble.png")
print(" - classification_report_heatmap.png")
print(" - genre_classes.npy")