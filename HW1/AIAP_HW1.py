import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
import numpy as np

# 데이터셋 로드
df = pd.read_csv('diamonds.csv')

# 범주형 변수 확인
categorical_columns = df.select_dtypes(include=['object']).columns

# 3.1.1 Ordinal Encoding
ordinal_encoder = OrdinalEncoder()
df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])

# 3.1.2 One-Hot Encoding
one_hot_encoder = OneHotEncoder(sparse=False)
cut_encoded = one_hot_encoder.fit_transform(df[['cut']])
print("One-Hot Encoded 'cut' variable's first 5 samples:\n", cut_encoded[:5])

# 연속형 변수 확인
continuous_columns = df.select_dtypes(include=[np.number]).columns

# 3.2.1 MinMaxScaling
min_max_scaler = MinMaxScaler()
df[continuous_columns] = min_max_scaler.fit_transform(df[continuous_columns])
print("\nFirst five rows after MinMaxScaling:\n", df.head())

# 데이터셋 다시 로드하여 StandardScaling 수행
df = pd.read_csv('diamonds.csv')
# 3.1.1 Ordinal Encoding 다시 수행 (StandardScaling에 앞서 필요)
df[categorical_columns] = ordinal_encoder.fit_transform(df[categorical_columns])

# 3.2.2 StandardScaling
standard_scaler = StandardScaler()
df[continuous_columns] = standard_scaler.fit_transform(df[continuous_columns])
print("\nFirst five rows after StandardScaling:\n", df.head())
