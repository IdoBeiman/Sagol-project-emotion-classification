import pandas as pd
from praatio import textgrid
import json


def open_grid(textgrid_path):
    tg = textgrid.openTextgrid(textgrid_path, includeEmptyIntervals=True)

    word_tier = tg.tierDict['words']
    df = pd.DataFrame([(start, end, label) for (start, end, label) in word_tier.entryList],
                      columns=['start', 'end', 'label'])

    return df


def divide_to_segments(df, segments):
    segments_list = []
    for segment in segments:   # assuming we have the end-of-segment time stamp
        seg_df = df[df['start'] <= segment]
        df = pd.concat([df, seg_df]).drop_duplicates(keep=False)
        segments_list.append(seg_df)

    return segments_list

def df_divided_to_segments(segments_list):
    df_segments = pd.DataFrame(columns=['start', 'end', 'text'])
    # for i in range(44):
    for i in range(len(segments_list)):
        start = segments_list[i]['start'].iloc[0]
        end = segments_list[i]['end'].iloc[-1]
        text = segments_list[i]['label'].tolist()
        df_segments = df_segments.append({"start": start, "end": end, "text": text}, ignore_index=True)
    return df_segments


def save_df(df, output_path):
    df.to_csv(output_path, index=False, header=False)


def get_segments(json_path):
    times = []
    f = open(json_path)
    data = json.load(f)
    for t in data['timestemps']:
        times.append(t)
    times.append(data['audio_length_sec'])
    return times


if __name__ == '__main__':
    file_name = 'wax_2_seg'
    text_grid = 'wax_2.TextGrid'
    json_file_name = 'config_exp12.json'
    tg = open_grid(text_grid)
    segments = get_segments(json_file_name)
    #list = divide_to_segments(tg, [2.9, 5.17, 9.6])
    list = divide_to_segments(tg, segments)
    final = df_divided_to_segments(list)
    save_df(final, '{}.csv'.format(file_name))