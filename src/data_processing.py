import pandas as pd
from sklearn.preprocessing import LabelEncoder

def preprocess_data(df):
    df = df.copy()
    df['TransactionStartTime'] = pd.to_datetime(df['TransactionStartTime'])
    df['hour'] = df['TransactionStartTime'].dt.hour
    df['day'] = df['TransactionStartTime'].dt.day
    
    le = LabelEncoder()
    for col in ['ProductCategory', 'ChannelId', 'PricingStrategy']:
        df[col] = le.fit_transform(df[col].astype(str))
    
    features = ['Amount', 'Value', 'hour', 'day', 'ProductCategory', 'ChannelId', 'PricingStrategy']
    X = df[features].fillna(0)
    return X