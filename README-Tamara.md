#What kind of scripts can be found here?
The scripts directory contains four subdirectories for different purposes:
1. **Label Collection**: contains the code for the web-based annotation tool as well as the locally-accessible tool which allows for both annotation and segmentation of files.
2. **Extract Annotation and Preprocess for ML**: contains scripts for extracting annotation results from the firestore and then restructuring them as ML input
3. **Audio Transcription**: containing scripts for transcribing the audio files into text
4. **Annotation Analysis and ML**: contains scripts for creating linear models and basic neural network to study the dataset and create predictions (and also for simple performance evaluation of the models)

#Label collection

##Web tool

###Create a new web annotation experiment
A copy of the code of “experiment-template” is found at scripts>Label Collection>Web-annotation-tool-template.
The setup of the experiment in the firebase is detailed in [this](https://docs.google.com/document/d/1vvXmXNp_33sAPYwnO2vrHwpfunoMgdPk9-16aJOUrSc/edit) document by Itamr. 

**Requirements:**
To edit the experiment: None! Any text editor will do
To deploy the edited files: firebase-cli

**Edit the experiment:**
The code to the experiment’s pages is at the public directory.
In this context, experiment means a bunch of story annotations that are published at the same time. Between stories the instructions or labels might be different, but within a story all the annotators are facing the exact same task.

**Consent form:** Edit it using the **index.html** file. The file is written in plain html. Edit the content of the paragraph tag to change the text.
**Instructions:** Edit both:
    - **Jspsych-instructions.html**: first instructions that follow the consent form. Written using JSPsych lib so use it’s objects to edit. Specifically, edit the variable `instructions` which is a JSPsych instruction type, and the content of this instance of the variable will be saved to the metadata file. 
    - **Instructions_remider.html**: the main experimental session (the annotation screen) contains a “instructions reminder” button that leads to this page. This page is written in plain html so edit the content of the paragraph tag to change the text. If the annotator viewed this page during their session, the content of this page will also appear in this subject’s metadata file.
**Debriefing:** Edit **feedback.html** which is written using JSPsych lib so use it’s objects to edit. Specifically, edit the variables `survey_page1`, ..., `survey_page4` which are different JSPsych survey types, and the content of these instances of the variables will be saved to the metadata file. Alternatively, create your own survey variables and just edit the metadata saving line: `meta["debrief_log"] = [<my_survay_variable_1>, <my_survay_variable_2>,...]`. To use more question types than currently available, download the relevant jspsych plugin and put it at public>js>jspsych_plugins.
**Annotation section:** This section is composed of the **annotation_section.html** file as well as the **annotation_section.js** file. To control this part create a new configuration file (will be explained later) and place it at public>configs. 
**Stories:** the audio files of the stories you wish to play should be placed in public>media.

**Create a configuration file:**
All configuration files must be stored at public>configs.
The files should be in json format, and their name should start with the prefix “config_exp”. 
Even in the same experiment, each story should have its own config file. 

Fields (Make sure to use the exact names, case sensitivity and typos included!):
- `audio_clip_name`: String. Full name of audio file the subject should annotate. For example: `"Revision_Quest.mp3"`
- `timestemps`: Float array. Containing the times (in seconds) of all the ofstets of the segments we wish to annotate. In other words, all the time points in which the audio should stop and an annotation box should appear. For example: `[10.83, 17.1, 27.91, 37.75, 46.45]`. This array doesn’t need to include the end of the audio file as it would stop there anyways.
- `questions`: An array of questions in a very specific format that will be explained in the next section. 
- `question_number`: Integer. Number of questions the subject will receive in each stop. In other words, the length of the “questions” array.
- `htmls`: dictionary. Matches each experiment section (except the main annotation part) to an html file. The keys are: “debriefing_html”, “instructions_html” and “instructions_reminder_html”. The files should reside in the “public/common” directory. Example: `{“debriefing_html":"feedback.html", instructions_html":"jspsych-instructions.html", “instructions_reminder_html":"instructions_remider.html"}`. This can be useful if you wish to use several versions of the instructions in a single experiment.
- `Experiment_version`: String. This will be the name of the firestore collection in which the results will be saved. Several stories can be included in the same experiment (meaning, several configuration files can and likely will contain the same value in this field). 

Questions format:
There are two types of questions: multiple choice questions and slider questions.
Multiple choice: dictionary with the following keys:
- `number`: unique string, preferably of an integer. This will mark the html object linked with this question.
- `tag`: string. This will be the tag of this question with regard to response collection (in the final dataset, each time section will contain this tag along with the response given to this question at this segment).
- `type`: String. should be exactly “multiple_choice”
- `text`: String. This is the text the subject will see above this question
- `min_label`: String. This is a text that will appear above the possible responses (right side). To keep it blank simply choose an empty string (“”).
- `middle_label`: String. This is a text that will appear above the possible responses (in the middle). To keep it blank simply choose an empty string (“”).
- `max_label`: String. This is a text that will appear above the possible responses (left side). To keep it blank simply choose an empty string (“”).
- `ans_num`: String of an integer. Should be exactly the length of the following “ans” array.
- `ans`: An array of strings. These strings will appear as the text of the choices for this question. The strings should be sorted as you wish to see them (first item will correspond to the most left button).
- `Ans_type`: there are two options only: “numeric” if all the possible responses are numbers, or “textual” otherwise. This will only affect the layout of the response buttons (wider buttons for numeric values).
Example 1:
`{"number": "1", "tag": "unpleasantness", "type": "multiple_choice", "text": "Unpleasantness", "min_label": "Not at all", "middle_label": " ", "max_label": "Very much", "ans_num": "5", "ans": ["1", "2", "3", "4", "5"], "ans_type": "numeric"}`

Example 2:
`{"number": "1", "tag": "unpleasantness", "type": "multiple_choice", "text": "Unpleasantness", "min_label": "", "middle_label": " ", "max_label": "", "ans_num": "5", "ans": ["Not at all", "maybe", "somewhat", "yes", "Very much"], "ans_type": "textual"}`

Slider: dictionary with the following keys:
- `number`: same as multiple choice
- `tag`: same as multiple choice
- `type`: String. should be exactly “slider”
- `text`: same as multiple choice
- `min_label`:same as multiple choice
- `middle_label`: same as multiple choice
- `max_label`: same as multiple choice
Note that regardless of the labels you give the edges, the actual values of the leftmost choice would be 0 and of the rightmost choice would be 100.

Example:
`{"number": "4", "tag": "control", "type": "slider", "text": "Control", "min_label": "Lack of control", "middle_label": " ", "max_label": "High control"}`

###Experiment data structure in the firestore
Each experiment will be composed of several stories. Each story has its own config file. All the data from a certain experiment will be saved as a collection with the experiment’s name. In the collection you’ll find a single document named “stories”. This document contains subcollections for each story in the experiment. Each story collection contains a document for each subject who annotated that story (even if the annotation wasn’t completed), named after this subject’s participation code. The document of each subject contains all the data relevant for that subject: annotations, answers to debriefing questions, and metadata (time of participation, logs for all the screens this subject has seen, the config file used for that subject, etc.)

###Link to the experiment
***https://experiment-template.web.app/?config=<number of  config file>.***
***https://experiment-template.web.app/?config=999*** will initiate an experiment based on the config file named **“config_exp999.json”** and ***https://experiment-template.web.app/?config=hi***
will initiate an experiment based on the config file named “config_exphi.json”.

###How to extract the answers from the firestore
Use the scripts under scripts>Extract Annotation and Preprocess for ML (will be explained later in this file)

###Possible errors
When we used this tool for the seminar, it seemed that for some subjects some stopping points were skipped. We did not manage to recreate the error ourselves so we don’t know what caused it, but here some changes were made to deal with the error. First, the annotation section code was dramatically changed to be simpler and work more efficiently. Second, the way the annotations were saved was changed so that if such error does reoccur, it will be spotted and will not affect the rest of the data by that annotator: The responses of each annotator is a list of dictionaries - a dictionary per stop with the following keys: 
- target_time: the time in which the stop was expected to occur based on the config file
- actual_time: when the stop actually occurred. Because of browser limitations we can’t actually stop in the target time, each run the stopping time for a specific target time will be slightly different (within 250ms range).
- A key for each label, that it’s value would be the annotation given to this label at this stop
When all the responses to a story are extracted, they are compared with the target time from the config file. If a dictionary for a certain target_time is missing, then it would still appear, but with null values on all the labels. In this way, if a stop is skipped, this annotor will still be usable for their other responses.



##Local tool
This tool can be used both for annotation and for segmentation of audio files locally. Meaning, the files are not uploaded anywhere so even confidential files can be used. It is constructed of front-end code only with no server, so occasionally some weird behaviors occur. However, for the most part it works just fine.
The annotation section is based on the code of the web tool, meaning it was last updated in April 2021. The segmentation section though, was not updated since the workshop (June 2020) since it would probably be replaced soon: it works, but the code behind it is very different so it might be less accurate with timings. 

###Requirements:
No requirements. Can be both used and edited using notepad and a browser only.

###Usage

**Annotation**
- Prepare a configuration file based on the instructions for the web-tool.
- Place the file you wish to annotate in the media directory
- Click “Annotation and segmentation tool”. This will open a browser but don’t worry, this still works locally, the files are safe!
- Click the “choose file” button, browse to your configuration file, and choose it.
- Click the “Annotate” button. This will lead you to the annotation screen that looks just like the web tool’s screen and worlds the same
- Annotate. Once you’ve finished you’ll be led to the end-screen.
- This section contains debriefing questions. Answer all of them and click “done”. 
This will prompt the tool to download 3 files: a copy of the config file you worked with (for logging purposes), the responses to the debrief and the responses to the annotations. Note that some browsers will first ask you to agree to the download of multiple files to your pc.

Error handling in the end-screen
Note that the annotations you provided in this section are saved as a “session variable”. This means that as long as you don’t close the browser or start a new annotation session, they should still be available to you at the “end-screen”. To retrieve the annotation:
- Open the console (F12)
- Type  `var AnnotationData = localStorage.getItem('Annotations');`
- Type `const annot = JSON.parse(AnnotationData).Annotations;`
- Copy the answer to a notepad.

**Segmentation**
- Prepare a configuration file based on the instructions for the web-tool. This configuration file can be lacking. It only really needs to include a direction to the audio file and a partial or empty timestamp list.
- Place the file you wish to annotate in the media directory
- Click “Annotation and segmentation tool”. This will open a browser but don’t worry, this still works locally, the files are safe!
- Click the “choose file” button, browse to your configuration file, and choose it.
- Click the “edit timestamps” button. This will lead you to the segmentation section. If your configuration file already included some time points they will appear on the white box. Otherwise, this box will be empty at the beginning.
- Play the audio and click the “stamp” or “s” key whenever you wish to add the current time point to the list. Play and pause the audio using the play/pause button on the screen or the space key. Click on a specific time point to edit it. Once you’ve clicked a time point the audio will continue from that point. You can delete the current time point (the one that appears in bold) by clicking the “delete” button or the backspace. Edit the time of the current time point by typing the new value in the replace box and clicking “replace”. Note that you can only replace at a valid time (greater than the previous time point and smaller than the next time point)  and invalid replacements simply will not respond. 
- At any time click the “save” button to download the updated configuration file that will now also include the time points you added.



#Extract Annotation and Preprocess for ML

##Annotation Extraction:
Use the **“extract_experiment_results.py”** script, and edit the “researcher parameters” section.

###Requirements:
Firebase_admin (including a credential file), numpy, pandas. On the T-17 will run under the environment “itamar-nlp”

###Inputs:
- `experiment_name`: Name of collection from the firebase you wish to extract
- `Subjects_to_remove`: path to a text file containing the subject codes of the subjects we wish to not extract their data (comma separated) 
- `Export_dir`: Path for the directory to which we wish to export the data. 
- `Extract_continious`: Boolean parameter stating if you wish to extract subjects’ continuous annotations
- `Extract_metadata`: Boolean parameter stating if you wish to extract subjects’ metadata
- `Extract_debreif`: Boolean parameter stating if you wish to extract subjects’ debriefing responses
- `Firebase_credentials_file`: path to the firebase credentials file

###Script’s Assumptions: 
- All subjects with same story ran at the same time/with regard to the same config file and debriefing questions (some parts of the code loads the config and debriefing structure for the first subject and apply it to all subjects of this story)
- The input dir contains a text file named 'subjects_to_remove.txt' which contains the codes of all the subjects we wish to skip (comma separated)

###Output:
The export folder will now include a folder for each story in this experiment. Each folder will contain:
- If `Extract_metadata==True`: “metadata_and_logs” folder that contains a json file per subject, that contains the metadata for this subject.
- If `Extract_debreif==True`: <story_name>_debreif.csv  file that contains a column per subject and a row for each question in the bebreif. The questions will be identified by their jspstch labels rather than their text.
- If `Extract_continious==True`: a <subject_code>_continuous.csv per subject. Containing a column per label and a row per time-point. Note that the last time point is not a number, but the text “infinity” .
	

##Prepare data for ML:
Use the **"create_ML_input.py"** script and edit the “researcher parameters” section. The ML section requires 3 types of data for each section:
- A transcription of the audio segment
- An embedding of that transcription (which will be used as features)
- An aggregated score of the subject’s annotation for a specific sentiment (which will be used as label)

###Requirements:
Pandas, pytorch and transformers python packages. 
A pre trained bert model for the embedding section ([for more info](https://huggingface.co/transformers/pretrained_models.html)). 

###Usage:
Change the following variables based on your needs:
- `Subjects_to_remove`: path for a text file containing the codes of the subjects you wish to exclude (comma separated)
- `export_dir`: Path for the directory to which we wish to export the data. 
- `transcription_dir`: Path of the location of the transcribed audio files. The expected format per story is word-by-word transcription. Meaning, each story needs to have a single csv file with 3 columns: word onset, word offset, word. All the words are arranged in a chronological order.
- `annotation_dir`: path of the output of the “extract_experiment_results.py” script (this script only requires the continues annotation files)
- `pretrained`: a path to the pretrained ‘transformers-huggingface’ model we wish to use for embedding.
- `sentiment`: the tag of the sentiment you wish to predict (corresponds to the “tag” of a question in the config file for an experiment).
- `drop_single_subject_data`: boolean stating if you wish the final result won't or will include the responses of the individual subjects

**To add new aggregation methods:**
Edit the “add_aggregated_scores” method found in this script. This method receives a df per single story: a column per subject, segment’s offset (timepoint) and that story’s name and needs to export the same df, just with the addition of aggregated scores columns.

###Output:
The output file will be exported to “export_dir” with the name “<sentiment>_ML_input.csv” and it includes the full data set as required by the ML scripts. This file contains a row for each segment in each of the stories in this experiment (chronological order within story). The columns are: timepoint (the offset of this segment from the beginning of the story), column for aggregated data to be used as labels, the segment’s text, and the embedding of that segment. 




#Audio Transcription
*Itamar should have a newer version of this code (after his adjustments) locally.*

The transcription is performed using google-cloud-speech-API, which receives files in mono flac format.    

## Setup
- In order to use the service, first obtain an account key file, and save it at *.Code*   
- In order to use the conversion script, download __ffmpeg__, and make sure you have the file __ffmpeg.exe__ in the path *./Prepare_Audio_for_Transcription/ffmpeg/bin*.   
- Go to __consts.py__ in *./Code* and edit the variavble "APPLICATION_ROOT" to be the local path of this folder

## How to perform transcription using this code
1. Put your mp3 files in *./Prepare_Audio_for_Transcription/in*
2. In *./Prepare_Audio_for_Transcription*, run __create_conversion_BAT_file.py__ to create a batch file of commands for ffmpeg.
	+ This script contains a hard-coded extension of "mp3". If your input is a "wav" file, edit accordingly.   
	+ This will create a batch file for all the files in the *./Prepare_Audio_for_Transcription/in* folder.   
3. In *./Prepare_Audio_for_Transcription*, run __Convert_all.BAT__ to convert from mp3 to mono-flac format.    
4. The converted should be found at *./Transcription_Input*   
5. In *./Code*, run __convert.CMD__. Keep in mind that if there's already a transcription output with the same name as one of your input files, this file will be skipped.   
6. The final transcription of __<audio name>.mp3__ should be found at *./Transcription_Output/<audio_name>*. The output files are:
	+ __<audio name>_full_text.txt__: The full transcribed text
	+ __<audio name>_word_list.csv__: A row for every word in the transcribed text (by order of appearance). Each row contains: {word, word onset, word offset}
	+ __<audio name>_automated_timestamps.json__: A suggestion for sentence segmentation, based on the punctuation provided in the transcription. The format of this file is suitable as an input for the manual segmentation tool.




#Annotation Analysis and ML
To use the scripts here you first need to prepare your data using the scripts in the “Extract Annotation and Preprocess for ML” section.

##Requirements:
Pandas, seaborn, sklearn, keras and tensorflow. All are installed in the itamar-nlp environment.

##Usage: 
First edit **"constant.py"** to suit your needs:
- `LABELLING_METHOD`: which aggregation metric should be used as the label of each sample. This should correspond to one of the columns in the data file (new aggregation methods can also be added by editing the “create_ML_input” script and re-running it).
- `Feat_vec`: an array with the names of all the columns from the data file that you wish to use as features.
- `prdicted_sentiment`: the sentiment that is being used as label
- `podasts_for_train`: array of the names of the stories to use (without file extension), should match the names of the “audio names” columns in the data file.
- `Results_dir`: path for storing the results of the analysis
- `Data_path`: path to the location of the data files. 

Then run **"direct_model_compare.py"**. This script performs leave one out cross validation: Each parameter set is iterated over all the stories, each time using a different story as a validation set, and the rest as a training set. The overall score for this parameter set is the average score across the stories. 
The script tests five parameter sets: 
- Baseline (BL): Predicts the mode label value in the training set
- Linear: a Bayesian-Ridge regression model [α1−4: 10−6 , λ1−2: 10−6 , normalize by l2]
- Fully connected network (FC): Two fully connected layers, first of size 128 and second of size64. [dropout rate: 0.6, activation function: tanh]
- Unidirectional LSTM (uLSTM): Two unidirectional LSTM layers, first of size 32 and second of size 20. [dropout rate: 0.3, activation function: sigmoid]
- Bidirectional LSTM (biLSTM): Two bidirectional LSTM layers, both of size 16.
[dropout rate: 0.4, activation function: sigmoid]

The parameters of the models can be edited and new models can be constructed using the “aModel” class which is defined in this script (explained later).

The script uses the keras library for the modeling section and according to the data size and the network size might take a lot of time to run. 

##Output:
Each run of the code will create a new directory within the results directory, and its name will be the date and time at the beginning of the run. It will include a subdirectory with the name of the sentiment we are modelling by. This directory will also include:
- A log file with all the parameters used for this run as well as interim results.
- A “<story>_model_prediction.csv” file for all the predictions created by all the models when <story> was used as a test batch.
- A “direct_model_comparison.csv” file that contains a comparison between the errors of the models throughout the analysis.
- A plot of the errors made by all the models in the entire analysis.
- A plot per story with the predictions made by each model compared to the baseline.

##The “Amodel” class:
Defined at the direct_model_compare.py script and builds a two layered neural network from keras. It’s a simple class that doesn’t allow for much, but it does give a nice starting point and is easy to extend.

###Constructions:
`n1`: number of neurons in the first layer
`n2`: number of neurons in the second layer
`d_o:` dropout rate of the first layer (should be a valid input for [keras lstm layer](https://keras.io/api/layers/recurrent_layers/lstm/))
`ac_func`: activation function in both layers (should be a valid input for [keras lstm layer](https://keras.io/api/layers/recurrent_layers/lstm/))
`model_type`: can be one of 3 strings: 'dense', 'uniLSTM' or 'biLSTM'. The first will create a network with 2 fully connected layers, the second with two unidirectional lstm layers and the third with two bidirectional lstm layers.
`name`: the name we want for this model in the plots and results and etc.
