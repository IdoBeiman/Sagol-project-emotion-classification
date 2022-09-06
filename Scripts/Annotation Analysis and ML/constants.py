
import os

# names of all the features in the full data dataframe
feat_vec = [str(i) for i in range(0, 768)]

# which sentiment are we learning on
predicted_sentiment = "Nostalgia"

model_types = ["dense","uniLSTM","BiLSTM"]

all_podcasts = ['train_activations_layer_8_sub005_operation_last_word_origin_full']# activations input file names
podasts_for_train = all_podcasts


root = os.path.join(os.getcwd(),"\debug") 

# results dir
Results_dir = f"{root}/Results"

# path of full data files (output of 'create_ML_input)
data_path = "./Tokenized_data"

all_emotions = ['Admiration', 'Adoration', 'Aesthetic appreciation', 'Amusement', 'Anger', 'Anxiety', 'Awe', 'Boredom',
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

balance_method = None # over for over-sampling, under for under-sampling, won't balance if None