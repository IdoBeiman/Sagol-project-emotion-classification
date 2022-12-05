import os
import json
import numpy as np
import pandas as pd
from praatio import textgrid
import re


def text_to_segments(sub_num):

    textGrid_input_dir = '/data/emotion_project/transcriptions/aligned/' + sub_num  # textGrid files
    timeStamps_input_dir = '/data/emotion_project/transcriptions/time_stamps/' + sub_num  # json files with segments time stamp
    output_segments_dir = '/data/emotion_project/transcriptions/episodes_to_segments/' + sub_num  # saving time stamped segments

    def open_grid(textgrid_path):
        tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)
        word_tier = tg.tierDict['words']
        df = pd.DataFrame([(start, end, label) for (start, end, label) in word_tier.entryList],
                          columns=['start', 'end', 'label'])
        return df

    def divide_to_segments(df, segments):
        # divides txt into segments using end-of-segment time stamps
        segments_list = []
        for segment in segments:
            seg_df = df[df['start'] <= segment]
            df = pd.concat([df, seg_df]).drop_duplicates(keep=False)
            segments_list.append(seg_df)
        return segments_list

    def df_divided_to_segments(segments_list):
        df_segments = pd.DataFrame(columns=['start', 'end', 'text'])
        for i in range(len(segments_list)):
            start = segments_list[i]['start'].iloc[0]
            end = segments_list[i]['end'].iloc[-1]
            text = ' '.join(segments_list[i]['label'].tolist()).strip()
            text = re.sub(r"\{[^{}]*\}", "", text)
            text = ' '.join(text.split())
            df_segments = df_segments.append({"start": start, "end": end, "text": text}, ignore_index=True)
        return df_segments

    def save_df(df, output_path, header=False):
        df.to_csv(output_path, index=False, header=header)

    def get_segments(json_path):
        # get end-of-segment time stamps
        times = []
        f = open(json_path)
        data = json.load(f)
        for t in data['timestemps']:
            times.append(t)
        times.append(data['audio_length_sec'])
        return times

    def csv_to_txt(file_path, output_path):
        df = pd.read_csv(file_path)
        df.drop(df.columns[[0, 1]], axis=1, inplace=True)
        values = [df.columns.values.tolist()] + df.values.tolist()
        np.savetxt(output_path, values, fmt='%s', newline='')

    if __name__ == '__main__':
        for filename in os.listdir(textGrid_input_dir):
            try:
                # open textGrid & time-stamps files and divide text into time-stamped segments
                file = filename.split('.TextGrid')[0]
                textGrid_path = os.path.join(textGrid_input_dir, filename)
                timeStamps_path = os.path.join(timeStamps_input_dir,
                                               f'{file}.json')  # assuming json files has the same name as textGrid files
                tg = open_grid(textGrid_path)
                ts = get_segments(timeStamps_path)
                segments_list = divide_to_segments(tg, ts)
                segments_df = df_divided_to_segments(segments_list)
                # saving df (time-stamped segments)
                output_path = os.path.join(output_segments_dir, f'{file}.csv')
                save_df(segments_df, output_path, True)
            except Exception as e:
                print(f'An Error Has Occurred: {e}')
