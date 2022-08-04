import os
import numpy as np
import pandas as pd
# gets csv file from textGrid_to_segments also get labeling tables (the labeled data), creates for each episode: start end text labels. for each subject: all segments with labels.

segments_input_dir = 'segments_input_dir'  # time stamped segments (csv files)
labels_input_dir = 'labels_input_dir'  # labeled segments (csv files)
ratings_per_episode_output_dir = ''   #
ratings_per_subject_file_path = ''  # path + file name for ratings per subject

emotions = ['Admiration', 'Adoration', 'Aesthetic appreciation', 'Amusement', 'Anger', 'Anxiety', 'Awe', 'Boredom',
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

def save_df(df, output_path, header=False):
    df.to_csv(output_path, index=False, header=header)


def csv_to_txt(file_path, output_path):
    df = pd.read_csv(file_path)
    df.drop(df.columns[[0, 1]], axis=1, inplace=True)
    values = [df.columns.values.tolist()] + df.values.tolist()
    np.savetxt(output_path, values, fmt='%s', newline='')

def load_df(segments_path, labels_path):
    # adds the
    segments_df = pd.read_csv(segments_path)
    labels_df = pd.read_csv(labels_path)
    joined = segments_df.join(labels_df.drop(labels_df.iloc[:, :6], axis=1))
    return joined

if __name__ == '__main__':
    try:
        subject_df = pd.DataFrame()
        for filename in os.listdir(segments_input_dir):
            # open textGrid & time-stamps files and divide text into time-stamped segments
            segments_path = os.path.join(segments_input_dir, filename)
            labels_path = os.path.join(labels_input_dir, filename)  # also csv file
            df = load_df(segments_path, labels_path)    # joined df per segment (start, end, text, existing labels)
            # save df (optional):
            save_df(df, os.path.join(ratings_per_episode_output_dir, filename), True)
            subject_df = pd.concat([subject_df, df], axis=0)
        # one csv per subject, cols = [start, end, text, emotions...]
        save_df(subject_df, ratings_per_subject_file_path, True)

    except Exception as e:
        print(f'An Error Has Occurred: {e}')

