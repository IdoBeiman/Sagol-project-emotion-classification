import os

# **** PATHS ****

ML_INPUT_DIR = 'C:\\Users\\Yuli\\Documents\\Uni\\Sagol-project-emotion-classification\\Tokenized_data'

ROOT = os.path.join(os.getcwd(), "\debug")
INPUT_FOLDER = "/data/emotion_project/idomayayuli/codeFolder/source_code/scripts"
PRETRAINED_PATH = '/data/emotion_project/transcriptions/labels_with_text/pre_train_data/pre-trained-model'

# data proccessing paths
TEXTGRID_INPUT_DIR = '/data/emotion_project/transcriptions/aligned/'  # textGrid files
TIME_STAMPS_INPUT_DIR = '/data/emotion_project/transcriptions/time_stamps/'   # json files with segments time stamp
OUTPUT_SEGMENTS_DIR = '/data/emotion_project/transcriptions/episodes_to_segments/'   # saving time stamped segments
# **** CONFIGURATION ****

MODELS = ["SNN", "Linear", "Baseline"]

N_SPLITS = 3
CV_SPLIT_METHOD = 'StratifiedKFold' # possible options - 'GroupKfold', 'StratifiedKFold', 'Random'

# if CV_SPLIT_METHOD is StratifiedKFold, SMOOTH_LABELS must be False
SMOOTH_LABELS = False
FILTER_ONES = True
BALANCE_DATA = False
BALANCE_METHOD = None   # over for over-sampling, under for under-sampling, won't balance if None

EPOCH = 30
BATCH_SIZE = 1

# NOT SUPPORTED YET
MULTI = False

ALL_SENTIMENTS = ['Admiration', 'Adoration', 'Amusement', 'Anger', 'Anxiety', 'Awe', 'Boredom',
                  'Calmness', 'Confusion', 'Contempt', 'Contentment', 'Craving', 'Despair', 'Disappointment', 'Disgust',
                  'Embarrassment', 'Entrancement', 'Envy', 'Excitement', 'Fear', 'Gratitude', 'Guilt',
                  'Hope', 'Horror', 'Interest', 'Irritation', 'Jealousy', 'Joy', 'Nostalgia', 'Pleasure', 'Pride',
                  'Relief', 'Romance', 'Sadness', 'Satisfaction', 'Sexual desire', 'Surprise', 'Sympathy', 'Triumph',
                  'Expectedness', 'Pleasantness', 'Unpleasantness', 'Goal Consistency', 'Caused by agent',
                  'Intentional Action', 'Caused by Self', 'Involved Close Others', 'Control', 'Morality', 'Self Esteem',
                  'Suddenness', 'Familiarity', 'Already Occurred', 'Certainty', 'Repetition', 'Coping', 'Mental States',
                  'Others Knowledge', 'Bodily\Disease', 'Other People', 'Self Relevance', 'Freedom', 'Pressure',
                  'Consequences', 'Danger', 'Self Involvement', 'Self Consistency', 'Relationship', 'Influence',
                  'Agent vs.Situation', 'Attention', 'Safety', 'Approach', 'Arousal', 'Commitment', 'Dominance',
                  'Effort', 'Fairness', 'Identity', 'Upswing']
PREDICTED_SENTIMENTS = ["Contentment","Sadness","Joy","Excitement","Pleasure","Satisfaction"]
