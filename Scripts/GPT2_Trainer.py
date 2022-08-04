import io
import os
import torch
from tensorflow import keras
from tqdm.notebook import tqdm
from torch.utils.data import Dataset, DataLoader
from ml_things import plot_dict, plot_confusion_matrix, fix_text
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,TrainingArguments,Trainer, GPT2Config,GPT2Tokenizer,AdamW, get_linear_schedule_with_warmup,GPT2ForSequenceClassification)
# Set seed for reproducibility.
print
set_seed(123)
# Number of training epochs (authors on fine-tuning Bert recommend between 2 and 4).
epochs = 2
# Number of batches - depending on the max sequence length and GPU memory. For 512 sequence length batch of 10 works without cuda memory issues. For small sequence length can try batch of 32 or higher.
batch_size = 32
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = None
# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = 'gpt2'
# Dictionary of labels and their id - this will be used to convert. String labels to number ids.
labels_ids = {'notSadAtAll': 0, 'notSad': 1,'neutral':2,'sad':3,'verySad':4}
# This is used to decide size of classification head.
n_labels = len(labels_ids)
"""## **Helper Functions**
* **init()** where we read in the dataset and transform text and labels into numbers.
* **len()** where we need to return the number of examples we read in. This is used when calling len(PodcastSentimentDataset()).
* **getitem()** always takes as an input an int value that represents which example from our examples to return from our dataset. If a value of 3 is passed, we will return the example form our dataset at position 3.

There are two main parts of this Data Collator class:
* **init()** where we initialize the tokenizer we plan to use, how to encode our labels and if we need to set the sequence length to a different value.
* **__call__()** used as function collator that takes as input a batch of data examples. It needs to return an object with the format that can be fed to our model. Luckily our tokenizer does that for us and returns a dictionary of variables ready to be fed to the model in this way: `model(**inputs)`. Since we are fine-tuning the model I also included the labels.
**train(dataloader, optimizer_, scheduler_, device_)**
"""
class PodcastSentimentDataset(Dataset):
  def __init__(self, path, use_tokenizer):
    print("the path given is: "+path)
    if not os.path.isdir(path):
      raise ValueError('Invalid `path` variable! Needs to be a directory')
    self.texts = []
    self.labels = []
    # Since the labels are defined by folders with data we loop  through each label.
    for label in ['notSadAtAll', 'notSad','neutral','sad','verySad']:
      sentiment_path = os.path.join(path, label)
      # Get all files from path.
      files_names = os.listdir(sentiment_path)
      # Go through each file and read its content.
      for file_name in tqdm(files_names, desc=f'{label} files'):
        file_path = os.path.join(sentiment_path, file_name)
        # Read content.
        content = io.open(file_path, mode='r', encoding='utf-8').read()
        # Fix any unicode issues.
        content = fix_text(content)
        # Save content.
        self.texts.append(content)
        # Save encode labels.
        self.labels.append(label)
    # Number of exmaples.
    self.n_examples = len(self.labels)
    return

  def __len__(self):
    return self.n_examples

  def __getitem__(self, item):
    r"""Given an index return an example from the position."""
    return {'text':self.texts[item],
            'label':self.labels[item]}

class Gpt2ClassificationCollator(object):
    r"""Data Collator used for GPT2 in a classificaiton task. 
    Arguments:
      use_tokenizer (:obj:`transformers.tokenization_?`):
          Transformer type tokenizer used to process raw text into numbers.
      labels_ids (:obj:`dict`):
          Dictionary to encode any labels names into numbers. Keys map to 
          labels names and Values map to number associated to those labels.
      max_sequence_len (:obj:`int`, `optional`)
          Value to indicate the maximum desired sequence to truncate or pad text
          sequences. If no value is passed it will used maximum sequence size
          supported by the tokenizer and model.
    """

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):
        # Tokenizer to be used inside the class.
        self.use_tokenizer = use_tokenizer
        # Check max sequence length.
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        # Label encoder used inside the class.
        self.labels_encoder = labels_encoder
        return
    def __call__(self, sequences):
        r"""
        Arguments:
          item (:obj:`list`):
              List of texts and labels.
        Returns:
          :obj:`Dict[str, object]`: Dictionary of inputs that feed into the model.
        """
        texts = [sequence['text'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels':torch.tensor(labels)})
        return inputs

def train(dataloader, optimizer_, scheduler_, device_):
  r"""
  Train pytorch model on a single pass through the data loader.
  Arguments:
      dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.
      optimizer_ (:obj:`transformers.optimization.AdamW`):
          Optimizer used for training.
      scheduler_ (:obj:`torch.optim.lr_scheduler.LambdaLR`):
          PyTorch scheduler.
      device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.
  Returns:
      :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss].
  """
  global model
  predictions_labels = []
  true_labels = []
  total_loss = 0
  model.train()
  # For each batch of training data...
  for batch in tqdm(dataloader, total=len(dataloader)):
    # Add original labels - use later for evaluation.
    true_labels += batch['labels'].numpy().flatten().tolist()
    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
    # Always clear any previously calculated gradients before performing a backward pass.
    model.zero_grad()
    # The documentation for this a bert model function is here:  https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
    outputs = model(**batch)
    # The call to `model` always returns a tuple, so we need to pull the  loss value out of the tuple along with the logits. We will use logits later to calculate training accuracy.
    loss, logits = outputs[:2]
    total_loss += loss.item()
    # Perform a backward pass to calculate the gradients.
    loss.backward()
    # Clip the norm of the gradients to 1.0. This is to help prevent the "exploding gradients" problem.
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Update parameters and take a step using the computed gradient.
    optimizer_.step()
    # Update the learning rate.
    scheduler_.step()
    # Move logits and labels to CPU
    logits = logits.detach().cpu().numpy()
    # Convert these logits to list of predicted labels values.
    predictions_labels += logits.argmax(axis=-1).flatten().tolist()
  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)
  # Return all true labels and prediction for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss

def validation(dataloader, device_):
  r"""Validation function to evaluate model performance on a 
  separate set of data.
  Arguments:
    dataloader (:obj:`torch.utils.data.dataloader.DataLoader`):
          Parsed data into batches of tensors.
    device_ (:obj:`torch.device`):
          Device used to load tensors before feeding to model.
  Returns:
    :obj:`List[List[int], List[int], float]`: List of [True Labels, Predicted
        Labels, Train Average Loss]
  """
  global model
  predictions_labels = []
  true_labels = []
  total_loss = 0
  # Put the model in evaluation mode--the dropout layers behave differently
  model.eval()
  # Evaluate data for one epoch
  for batch in tqdm(dataloader, total=len(dataloader)):
    # add original labels
    true_labels += batch['labels'].numpy().flatten().tolist()
    # move batch to device
    batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
    # Telling the model not to compute or store gradients, saving memory and speeding up validation
    with torch.no_grad():        
        outputs = model(**batch)
        # The call to `model` always returns a tuple, so we need to pull the loss value out of the tuple along with the logits. We will use logits later to to calculate training accuracy.
        loss, logits = outputs[:2]
        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        total_loss += loss.item()
        # get predicitons to list
        predict_content = logits.argmax(axis=-1).flatten().tolist()
        predictions_labels += predict_content
  # Calculate the average loss over the training data.
  avg_epoch_loss = total_loss / len(dataloader)
  # Return all true labels and prediciton for future evaluations.
  return true_labels, predictions_labels, avg_epoch_loss
"""## **Load Model and Tokenizer**
Loading the three essential parts of the pretrained GPT2 transformer: configuration, tokenizer and model. 
In creating the `model_config` I will mention the number of labels I need for my classification task. Since I only predict two sentiments: positive and negative I will only need two labels for `num_labels`.
Creating the `tokenizer` is pretty standard when using the Transformers library. After creating the tokenizer it is critical for this tutorial to set padding to the left `tokenizer.padding_side = "left"` and initialize the padding token to `tokenizer.eos_token` which is the GPT2's original end of sequence token. This is the most essential part of this tutorial since GPT2 uses the last token for prediction so we need to pad to the left.
Since we have a custom padding token we need to initialize it for the model using `model.config.pad_token_id`.
Finally we will need to move the model to the device we defined earlier.
"""
print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_name_or_path, num_labels=n_labels)
print (model_config)
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token
print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path, config=model_config)
loss_fn = keras.metrics.RootMeanSquaredError()
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id
model.to(device)
print('Model loaded to `%s`'%device)

"""## **Dataset and Collator**
This is where I create the PyTorch Dataset and Data Loader with Data Collator objects that will be used to feed data into our model.
This is where I use the **PodcastSentimentDataset** class to create the PyTorch Dataset that will return texts and labels.
Since we need to input numbers to our model we need to convert the texts and labels to numbers. This is the purpose of a collator! It takes data outputted by the PyTorch Dataset and passed through the Data Collator function to output the sequence for our model.
I'm keeping the tokenizer away from the PyTorch Dataset to make the code cleaner and better structured. You can obviously use the tokenizer inside the PyTorch Dataset and output sequences that can be used straight into the model without using a Data Collator.
I strongly recommend to use a validation text file in order to determine how much training is needed in order to avoid overfitting. After you figure out what parameters yield the best results, the validation file can be incorporated in train and run a final train with the whole dataset.
The data collator is used to format the PyTorch Dataset outputs to match the inputs needed for GPT2.
"""
# Create data collator to encode text and labels into numbers.
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, labels_encoder=labels_ids, max_sequence_len=max_length)
print('Dealing with Train...')
# Create pytorch dataset.
train_dataset = PodcastSentimentDataset(path='/Users/idobe/Desktop/podcastsMock/train', use_tokenizer=tokenizer)
print('Created `train_dataset` with %d examples!'%len(train_dataset))
# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
print('Created `train_dataloader` with %d batches!'%len(train_dataloader))
print('Dealing with Validation...')
# Create pytorch dataset.
valid_dataset =  PodcastSentimentDataset(path='/Users/idobe/Desktop/podcastsMock/test', use_tokenizer=tokenizer)
print('Created `valid_dataset` with %d examples!'%len(valid_dataset))
# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `eval_dataloader` with %d batches!'%len(valid_dataloader))

"""## **Train**"""
optimizer = AdamW(model.parameters(),
                  lr = 2e-5, # default is 5e-5, our notebook had 2e-5
                  eps = 1e-8 # default is 1e-8.
                  )
# Total number of training steps is number of batches * number of epochs. `train_dataloader` contains batched data so `len(train_dataloader)` gives us the number of batches.
total_steps = len(train_dataloader) * epochs
# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0,
num_training_steps = total_steps)
# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss':[], 'val_loss':[]}
all_acc = {'train_acc':[], 'val_acc':[]}
# Loop through each epoch.
print('Epoch')
for epoch in tqdm(range(epochs)):
  print()
  print('Training on batches...')
  # Perform one full pass over the training set.
  train_labels, train_predict, train_loss = train(train_dataloader, optimizer, scheduler, device)
  train_acc = accuracy_score(train_labels, train_predict)
  # Get prediction form model on validation data. 
  print('Validation on batches...')
  valid_labels, valid_predict, val_loss = validation(valid_dataloader, device)
  val_acc = accuracy_score(valid_labels, valid_predict)
  # Print loss and accuracy values to see how training evolves.
  print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f"%(train_loss, val_loss, train_acc, val_acc))
  print()
  # Store the loss value for plotting the learning curve.
  all_loss['train_loss'].append(train_loss)
  all_loss['val_loss'].append(val_loss)
  all_acc['train_acc'].append(train_acc)
  all_acc['val_acc'].append(val_acc)
# Plot loss curves.
plot_dict(all_loss, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
# Plot accuracy curves.
plot_dict(all_acc, use_xlabel='Epochs', use_ylabel='Value', use_linestyles=['-', '--'])
"""## **Evaluate**"""
true_labels, predictions_labels, avg_epoch_loss = validation(valid_dataloader, device)
# Create the evaluation report.
evaluation_report = classification_report(true_labels, predictions_labels, labels=list(labels_ids.values()), target_names=list(labels_ids.keys()))
# Show the evaluation report.
print(evaluation_report)
# Plot confusion matrix.
plot_confusion_matrix(y_true=true_labels, y_pred=predictions_labels, classes=list(labels_ids.keys()), normalize=True, magnify=0.1,)