from pathlib import Path


MAIN_PATH = Path('.')
TRAIN = MAIN_PATH / 'data' / 'Train.csv'
TEST = MAIN_PATH / 'data' / 'Test.csv'
SUBMISSION = MAIN_PATH / 'data' / 'SampleSubmission.csv'

N_FOLDS = 4
RANDOM_STATE = 1
THRESHOLD = 0.15

TARGET = 'target'

CATBOOST_PARAMS = {
    'random_state': RANDOM_STATE,
    'eval_metric': 'F1',
}
