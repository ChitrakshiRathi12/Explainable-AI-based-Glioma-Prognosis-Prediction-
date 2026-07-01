import pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# STEP 1: Load your dataset
df = pd.read_csv(r"C:\Users\Chitrakshi\Downloads\augmented_TCGA_GBM_LGG_Mutations_5x.csv")
target = "Grade"
features = ["IDH1", "ATRX", "TP53", "PTEN", "EGFR"]  # add your final features here

df[target] = df[target].astype(str)
X = df[features]
y = df[target]

# STEP 2: Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# STEP 3: Save train/test for reusability
train_df = X_train.copy()
train_df[target] = y_train
test_df = X_test.copy()
test_df[target] = y_test

# STEP 4: Train AutoGluon models
predictor = TabularPredictor(label=target).fit(
    train_data=train_df,
    time_limit=600,
    hyperparameters={
        'XGB': {},
        'GBM': {},
        'CNN': {},
        'RNN': {},
        'SVM': {}
    }
)

# STEP 5: Predict using AutoGluon best model
autogluon_preds = predictor.predict(test_df)

# STEP 6: Collect metrics
leaderboard = []
leaderboard.append({
    'Model': 'AutoGluon_Ensemble',
    'Accuracy': accuracy_score(y_test, autogluon_preds)
})