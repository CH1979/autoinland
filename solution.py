import logging

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold

import settings


logging.info('Reading data...')
train = pd.read_csv(settings.TRAIN, parse_dates=[
        'Policy Start Date',
        'Policy End Date',
        'First Transaction Date'
    ])
test = pd.read_csv(settings.TEST, parse_dates=[
        'Policy Start Date',
        'Policy End Date',
        'First Transaction Date'
    ])
sub = pd.read_csv(settings.SUBMISSION)

logging.info('Prepare features...')
features = list(train.columns)
features.remove('target')
features.remove('ID')
features.remove('Policy Start Date')
features.remove('Policy End Date')
features.remove('First Transaction Date')

cat_features = features.copy()
cat_features.remove('Age')
cat_features.remove('No_Pol')

train = train.fillna('NaN')
test = test.fillna('NaN')

for df in [train, test]:
    for n, col in enumerate([
        'Policy Start Date',
        'Policy End Date',
        'First Transaction Date'
    ]):
        df[f'year_{n}'] = df[col].dt.year
        df[f'month_{n}'] = df[col].dt.month
        df[f'day_{n}'] = df[col].dt.day
        df[f'weekday_{n}'] = df[col].dt.weekday

for n in range(2):
    features.extend([
        f'year_{n}',
        f'month_{n}',
        f'day_{n}',
        f'weekday_{n}'
    ])

logging.info('Define and training model...')
folds = StratifiedKFold(n_splits=settings.N_FOLDS)
model = CatBoostClassifier(**settings.CATBOOST_PARAMS)
scores = []

for i, (trn_idx, val_idx) in enumerate(folds.split(
    train[features],
    train[settings.TARGET]
)):
    print('-' * 30)
    print(f'Fold # {i}')
    model.fit(
        X=train.loc[trn_idx, features],
        y=train.loc[trn_idx, settings.TARGET],
        eval_set=[(
            train.loc[val_idx, features],
            train.loc[val_idx, settings.TARGET]
        )],
        verbose_eval=100,
        cat_features=cat_features
    )
    pred = model.predict_proba(train.loc[val_idx, features])[:, 1]
    pred = np.where(pred > settings.THRESHOLD, 1, 0)
    score = f1_score(
        train.loc[val_idx, settings.TARGET],
        pred
    )
    scores.append(score)
    sub[settings.TARGET] += model.predict_proba(
        test[features]
    )[:, 1] / settings.N_FOLDS
mean_score = np.mean(scores)

logging.info(f'Mean score on validation: {mean_score}')

logging.info('Saving result...')
sub['target'] = np.where(sub['target'] > settings.THRESHOLD, 1, 0)
sub.to_csv(settings.MAIN_PATH / 'subs' / f'cat-{mean_score:2.8}.csv', index=False)
