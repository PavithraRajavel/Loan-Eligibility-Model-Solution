from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def split_data(df):
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    return train_test_split(X, y, test_size=0.2, random_state=123)


def scale_data(X_train, X_test):
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def train_logistic_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def train_r_model(X_train, y_train):
    model = RandomForestClassifier(random_state=123)
    model.fit(X_train, y_train)
    return model


def cross_validate_rf(X_train, y_train):
    kfold = KFold(n_splits=5, shuffle=True, random_state=123)
    model = RandomForestClassifier(random_state=123)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)
    return scores

