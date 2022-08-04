import io
import os
path = '/Users/idobe/Desktop/podcastsMock'
for label in ['notSadAtAll', 'notSad','neutral','sad','verySad']:
    sentiment_path = os.path.join(path, label)
    # Get all files from path.
    files_names = os.listdir(sentiment_path)
    choosed = random.choices(files_names, k=3)
    for file_name in choosed:
        save_path = os.path.join(path, "/test/"+label)
        os.rename()
    # Go through each file and read its content.
    for file_name in tqdm(files_names, desc=f'{label} files'):