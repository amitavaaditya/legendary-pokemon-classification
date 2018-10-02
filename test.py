##=================================Template==============================
##===============
import psycopg2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import  Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, precision_score
import matplotlib.pyplot as plt
import redis
from multiprocessing import Process, Queue
import time


def db_init():
    conn = psycopg2.connect(database='chillindo', user='techno', password='')
    curr = conn.cursor()
    return conn, curr


def read_from_database(conn, curr):
    QUERY_SQL = 'SELECT * from pokemon'
    df = pd.read_sql(QUERY_SQL, conn, index_col='_id')
    return df


def preprocess_data(df):
    df.drop('name', axis='columns', inplace=True)
    all_types = df['type_1'].unique()
    for p_type in all_types:
        df[p_type] = 0
    df = df.apply(map_types, axis='columns')
    df.drop('type_1', axis='columns', inplace=True)
    df.drop('type_2', axis='columns', inplace=True)
    df['legendary'] = df['legendary'].astype('int64')
    X = df.drop('legendary', axis='columns')
    y = df['legendary']
    return X, y


def map_types(row):
    types = [
        'Grass', 'Fire', 'Water',
        'Bug', 'Normal', 'Poison', 'Electric', 'Ground', 'Fairy',
        'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice', 'Dragon', 'Dark',
        'Steel', 'Flying'
    ]
    type1 = row['type_1']
    type2 = row['type_2']
    for type in (type1, type2):
        if type in types:
            row[type] = 1
    return row


def split(X, y):
    return train_test_split(X, y, random_state=0, test_size=0.25, stratify=y)


def build_model():
    pipeline = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])
    param_grid = {
        'classifier__C': np.logspace(-3, -1, 10),
    }
    model = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=-1, verbose=2, refit=True, scoring='f1')
    return model


def train_and_validate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    print('Training accuracy: {}'.format(model.score(X_train, y_train)))
    print('Validation accuracy: {}'.format(model.best_score_))
    print('Test accuracy: {}'.format(model.score(X_test, y_test)))
    print('Best model params: {}'.format(model.best_params_))
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    for group in list(zip(fpr, tpr, thresholds)):
        print(group)
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Negative Rate')
    plt.title('ROC AUC score: {}'.format(roc_auc_score(y_test, y_prob)))
    plt.show()
    best_tpr = tpr[7]
    best_fpr = fpr[7]
    best_threshold = thresholds[7]
    y_pred = np.zeros(y_prob.shape[0])
    y_pred[y_prob > best_threshold] = 1
    precision = precision_score(y_test, y_pred)
    print('Precision: {}'.format(precision))
    return best_tpr, best_fpr, precision


def store_results(tpr, fpr, precision):
    r = redis.Redis(
        host='localhost',
        port=6379,
        password=''
    )
    r.set('tpr', tpr)
    r.set('fpr', fpr)
    r.set('precision', precision)
    print(r.get('tpr'))
    print(r.get('fpr'))
    print(r.get('precision'))


def process1(conn, curr, q):
    df = read_from_database(conn, curr)
    q.put(df)


def process2(q):
    df = q.get()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split(X, y)
    model = build_model()
    tpr, fpr, precision = train_and_validate(model, X_train, X_test, y_train, y_test)
    store_results(tpr, fpr, precision)


if __name__ == '__main__':
    conn, curr = db_init()
    q = Queue()
    p1 = Process(target=process1, args=(conn, curr, q))
    p2 = Process(target=process2, args=(q,))
    print('Starting p1')
    p1.start()
    print('10 secs delay')
    time.sleep(10)
    print('Starting p2')
    p2.start()
    print('waiting for p2 to finish')
    p2.join()
    print('p2 complete')
