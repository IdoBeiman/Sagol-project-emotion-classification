# Sagol-project-emotion-classification
 
# Configurations:
 1. First, create your own venv in your local development folder, use this guide if you are using vscode - https://code.visualstudio.com/docs/python/environments
 2. Install all the needed packages using the following command in your CLI: pip3 install -r Scripts/requirements.txt
 3. Check that all of the constants are defined to handle your local development env.
 4. Use main.py in Data Organization and get_activations.py to pre-process the data.
 5. Make sure you have some data in the Tokenized_data Folder placed under the repo root directory. 
 This folder should contain csv files with the relevant tokenization values created by the pre-processing scripts listed above.

# Pipeline:

1. Using the Data Organization folder scripts (main.py) create a dataframe containing the different segments with their respected emotional rankings
2. With the proccessed segments create model activations, using the get_activations.py script where you will be able to select the model, layer and computational operation to be used on the data. These scripts can be found in the Activations folder
3. The activations dataframes should be located in the ML_INPUT_DIR path defined in the constants file. Then, run the direct_model_compare.py script to get the model results for your proccessed data
# Possible setbacks:
 1. Import seaborn - python -m pip install seaborn


# Running model
 1. Use direct_model_compare.py to run the model and constants.py to configure it.
 2. Output will be saved in destination_folder.
