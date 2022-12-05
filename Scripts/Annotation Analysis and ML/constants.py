import os

# **** PATHS ****
# ML_INPUT_DIR = "./Tokenized_data"
ML_INPUT_DIR = 'C:\\Users\\Yuli\\Documents\\Uni\\Sagol-project-emotion-classification\\Tokenized_data'

ROOT = os.path.join(os.getcwd(), "\debug")

# **** CONFIGURATION ****
PREDICTED_SENTIMENTS = ['Sadness']

MODELS = ["SNN", "Linear", "Baseline"]

N_SPLITS = 3
CV_SPLIT_METHOD = 'StratifiedKFold'
# options - 'GroupKfold', 'StratifiedKFold', 'Random'
# if CV_SPLIT_METHOD is StratifiedKFold, SMOOTH_LABELS must be False
SMOOTH_LABELS = False
FILTER_ONES = True
BALANCE_DATA = False
BALANCE_METHOD = None   # over for over-sampling, under for under-sampling, won't balance if None

EPOCH = 6
BATCH_SIZE = 1

# NOT SUPPORTED YET
MULTI = False

ALL_SENTIMENTS = ['Admiration', 'Adoration', 'Aesthetic appreciation', 'Amusement', 'Anger', 'Anxiety', 'Awe', 'Boredom',
                  'Calmness', 'Confusion', 'Contempt', 'Contentment', 'Craving', 'Despair', 'Disappointment', 'Disgust',
                  'Embarrassment', 'Empathic pain', 'Entrancement', 'Envy', 'Excitement', 'Fear', 'Gratitude', 'Guilt',
                  'Hope', 'Horror', 'Interest', 'Irritation', 'Jealousy', 'Joy', 'Nostalgia', 'Pleasure', 'Pride',
                  'Relief', 'Romance', 'Sadness', 'Satisfaction', 'Sexual desire', 'Surprise', 'Sympathy', 'Triumph',
                  'Expectedness', 'Pleasantness', 'Unpleasantness', 'Goal Consistency', 'Caused by agent',
                  'Intentional Action', 'Caused by Self', 'Involved Close Others', 'Control', 'Morality', 'Self Esteem',
                  'Suddenness', 'Familiarity', 'Already Occurred', 'Certainty', 'Repetition', 'Coping', 'Mental States',
                  'Others Knowledge', 'Bodily\Disease', 'Other People', 'Self Relevance', 'Freedom', 'Pressure',
                  'Consequences', 'Danger', 'Self Involvement', 'Self Consistency', 'Relationship', 'Influence',
                  'Agent vs.Situation', 'Attention', 'Safety', 'Approach', 'Arousal', 'Commitment', 'Dominance',
                  'Effort', 'Fairness', 'Identity', 'Upswing']











# MODELS = ["SNN", "uniLSTM", "BiLSTM", "Linear", "Baseline"]

