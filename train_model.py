import pandas as pd
from sklearn.model_selection import train_test_split
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score  # Accuracy metrics
import pickle


def concat_all_csv(filepath):
    files = os.listdir(filepath)
    df = pd.DataFrame()
    for f in files:
        df_now = pd.read_csv(filepath + '/' + f)
        df = pd.concat([df, df_now])
    return df


if __name__ == '__main__':
    df = concat_all_csv('./dataset')

    X = df.drop('class', axis=1)  # features
    y = df['class']  # target value

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    # pipelines = {
    #     'lr': make_pipeline(StandardScaler(), LogisticRegression()),
    #     'rc': make_pipeline(StandardScaler(), RidgeClassifier()),
    #     'rf': make_pipeline(StandardScaler(), RandomForestClassifier()),
    #     'gb': make_pipeline(StandardScaler(), GradientBoostingClassifier()),
    # }
    pipelines = {
        'rf': make_pipeline(StandardScaler(), RandomForestClassifier())
    }

    fit_models = {}
    for algo, pipeline in pipelines.items():
        model = pipeline.fit(X_train, y_train)
        fit_models[algo] = model

    for algo, model in fit_models.items():
        yhat = model.predict(X_test)
        print(algo, accuracy_score(y_test, yhat))

    with open('body_language.pkl', 'wb') as f:
        pickle.dump(fit_models['rf'], f)
