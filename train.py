import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import datetime

vocab_size = 10000
embedding_dim = 16
max_length = 100 
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
training_size = 1000 # Change this to number of final samples
labels = ['Hi person1', 'Hi person2'] # Change to your name and other person's name

# Lists to store messages 
person1 = [] 
person2 = []

# Load JSON file and add each message to appropriate list
with open('message_1.json') as json_file: 
	data = json.load(json_file)
	for p in data['messages']:
		content_present = p.get('content')
		if content_present == None:
			continue
		if p['sender_name'] == 'person1': # change 'person1' to your name
			person1.append(p['content'])
		else:
			person2.append(p['content'])

# Make label lists
person1_labels = []
person2_labels = []
for _ in person1: person1_labels.append(1)
for _ in person2: person2_labels.append(0)

# combine messages lists and labels lists, these will be shuffled when training
messages = person1 + person2 
y = person1_labels + person2_labels

# fit tokenizer on messages
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(messages)
word_index = tokenizer.word_index
training_sequences = tokenizer.texts_to_sequences(messages)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Need this block to get it to work with TensorFlow 2.x
training_padded = np.array(training_padded)
training_labels = np.array(y)

# Define and compile model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# For TensorBoard
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# train
history = model.fit(training_padded, training_labels, epochs=4, verbose=2,
					 shuffle=True, batch_size=10, callbacks=[tensorboard_callback])


# Predict with your model! :)
print('Type exit to exit!')
done = False
while(not done):
	test_message = input('Type in a message: ')
	if test_message == 'exit':
		done = True
	sequences = tokenizer.texts_to_sequences([test_message])
	padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
	result = model.predict(padded)
	if(result >= 0.5):
		print(labels[0])
	else:
		print(labels[1])