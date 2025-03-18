from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_model(df, use_ridge=False):
    """Treina o modelo de regressão"""
    X = df[['age', 'gender', 'bmi', 'no_of_children', 'smoker', 'region']].astype(float)
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, random_state=1)

    # Escolha entre Ridge e Linear Regression
    if use_ridge:
        model = Ridge(alpha=0.1)
    else:
        model = Pipeline([
            ('scale', StandardScaler()),
            ('polynomial', PolynomialFeatures(include_bias=False)),
            ('Model', LinearRegression())
        ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    score = r2_score(y_test, y_pred)

    return model, y_pred, y_test
