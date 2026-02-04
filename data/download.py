import pandas as pd
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target

df.to_csv('cancer_data.csv', index=False)
print("Breast cancer dataset saved to cancer_data.csv")