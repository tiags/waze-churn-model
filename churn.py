import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import statsmodels.api as sm


def assign_risk(prob):
    if prob > 0.7:
        return 'High Risk'
    elif prob > 0.4:
        return 'Medium Risk'
    else:
        return 'Low Risk'

df = pd.read_csv('~/waze_retention.csv')

for col in ['sessions', 'drives', 'total_sessions', 'n_days_after_onboarding', 'total_navigations_fav1',	'total_navigations_fav2',
            'driven_km_drives',	'duration_minutes_drives',	'activity_days',	'driving_days', 'km_per_driving_day', 'percent_sessions_in_last_month',
            'total_sessions_per_day',	'km_per_hour',	'km_per_drive',	'percent_of_drives_to_favorite']:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df['Retention'] = df['label2'].map({1: 0, 0: 1})
X = df.drop(columns=['customerID', 'device', 'label', 'label2', 'Retention'])
y = df['label2']  # 0 = retained, 1 = churned

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

clf = RandomForestClassifier(class_weight='balanced')
clf.fit(X_train, y_train)

y_probs = clf.predict_proba(X_test)[:, 1]
for t in np.arange(0.2, 0.9, 0.1):
    preds = (y_probs > t).astype(int)
    print(classification_report(y_test, preds))

print(classification_report(y_test, clf.predict(X_test)))

df['Churn_Prob'] = clf.predict_proba(X)[:, 1]

#OLS
X_ols = sm.add_constant(X.astype(float))  
ols = sm.OLS(df['Churn_Prob'], X_ols).fit()
ols_summary = ols.summary()
print(ols_summary)
ols_df = pd.DataFrame({
    'Feature': ols.params.index,
    'Coefficient': ols.params.values,
    'P_Value': ols.pvalues.values
})
ols_df['Abs_Coefficient'] = ols_df['Coefficient'].abs()
ols_df = ols_df.sort_values(by='Abs_Coefficient', ascending=False)
ols_df.to_csv("ols_retention_insights.csv", index=False)

#Retention CSV
df['PredictedChurn'] = (df['Churn_Prob'] > 0.4).astype(int)
df['RiskTier'] = df['Churn_Prob'].apply(assign_risk)
print(df[['customerID', 'Churn_Prob', 'PredictedChurn', 'Retention', 'RiskTier']].head())
df[['customerID', 'Churn_Prob', 'PredictedChurn', 'Retention', 'RiskTier']].to_csv("waze_retention_results.csv", index=False)

importances = clf.feature_importances_
features = X.columns
feat_df = pd.DataFrame({'Feature': features, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=False)

# Plot top 10
feat_df.head(10).plot(kind='barh', x='Feature', y='Importance', legend=False)
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances")
plt.tight_layout()
plt.show()

