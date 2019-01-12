import tensorflow as tf
import nltk
import collections
import numpy as np
from tensorflow.contrib import rnn

TEXT_FILE_NAME = 'atom_text'
NUM_WORDS_FOR_PREDICTION = 3
tf.reset_default_graph()

# Function to take a list of words and create a dictionary from them
# Takes in a list of strings where each word is its own string

def embed_words(word_list):
    vocab = collections.Counter(word_list).most_common()
    vocab_dict = dict()
    for word, _ in vocab:
        vocab_dict[word] = len(vocab_dict)
    reverse_vocab_dict = dict(zip(vocab_dict.values(), vocab_dict.keys()))
    return vocab_dict, reverse_vocab_dict


# Function to help build an input data set and corresponding labels
# Takes in a list of words where each word is its own string
# Turns each string into a number (index in the dictionary) and sorts into groups of 3 for prediction
# Outputs the list of input words (as numbers) and their corresponding labels (also numbers)
def build_data_set(word_list, vocab_dict):
    X = []
    Y = []
    sample = []
    for index in range(0, len(word_list) - NUM_WORDS_FOR_PREDICTION):
        for i in range(0, NUM_WORDS_FOR_PREDICTION):
            sample.append((vocab_dict[word_list[index + i]]))
            if (i + 1) % NUM_WORDS_FOR_PREDICTION == 0:
                X.append(sample)
                Y.append(vocab_dict[word_list[index + i + 1]])
                sample = []
    return X, Y


def interpret_results(results):
    list_of_predictions = []
    for result in results:
        max_index = np.argmax(result, 0)
        predicted_word = reverse_vocab_dict[max_index]
        list_of_predictions.append(predicted_word)
    return list_of_predictions


# Function to build up the core of the machine learning model
# Takes the input (typically from a placeholder) and outputs the output after running input through the network
# RNNs are very useful for scoring word combinations and generating new text
# RNNs work on timesteps which essentially measure what path was taken x amount of steps ago and use this to make
# future predictions. We need to learn fewer params this way
def RNN(x_input):
    # Flatten input for better analysis
    x = tf.unstack(x_input, NUM_WORDS_FOR_PREDICTION, 1)
    # Create an LSTM cell, a basic unit (kind of like a neuron) of an RNN, we have 512 in our network
    lstm_cell = rnn.BasicLSTMCell(512)
    # Creates a network (think of this as a dense layer) made up of basic lstm cells
    outputs, _ = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    weights = tf.Variable(tf.random_normal([512, len(vocab_dict)]))
    biases = tf.Variable(tf.random_normal([len(vocab_dict)]))

    op = tf.matmul(outputs[-1], weights) + biases

    return op


# Read the text file
with open(TEXT_FILE_NAME) as f:
    text = f.read()
# Turn text into array of strings
word_list = nltk.tokenize.word_tokenize(text)
# Create the vocab dictionary
vocab_dict, reverse_vocab_dict = embed_words(word_list)

# Create training and testing data sets
x_train, y_train = build_data_set(word_list, vocab_dict)

# Inputs for training and testing (y_input just for training)
x_input = tf.placeholder(tf.float32, [None, NUM_WORDS_FOR_PREDICTION, 1], 'x_input')
y_input = tf.placeholder(tf.float32, [None, len(vocab_dict)])

# Feed input through the RNN
logits = RNN(x_input)
# Final output from our graph
y_output = tf.nn.softmax(logits, name='y_output')

# Loss, optimizer, and accuracy functions
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_input, logits=logits))
train_step = tf.train.RMSPropOptimizer(0.001).minimize(loss_op)
correct_prediction = tf.equal(tf.argmax(y_output, 1), tf.argmax(y_input, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

saver = tf.train.Saver()

# Create the tf session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    epochs = 200
    batch_size = 20

    tf.train.write_graph(sess.graph_def, '.', 'text_predictor.pbtxt', False)

    # Main training loop
    for i in range(epochs):
        last_batch = len(x_train) % batch_size
        num_train_steps = (len(x_train) / batch_size) + 1
        # Run the loop for each batch
        for step in range(int(num_train_steps)):
            # Get a batch of the training data and labels
            x_batch = x_train[(step * batch_size): ((step + 1) * batch_size)]
            y_batch = y_train[(step * batch_size): ((step + 1) * batch_size)]
            y_batch_encoded = []
            for y in y_batch:
                one_hot = np.zeros([len(vocab_dict)], dtype=float)
                one_hot[y] = 1.0
                y_batch_encoded = np.concatenate((y_batch_encoded, one_hot))
            x_batch = np.array(x_batch)
            y_batch_encoded = np.array(y_batch_encoded)
            # Reshape the batches for input into our model
            if len(x_batch) < batch_size:
                x_batch = x_batch.reshape(last_batch, NUM_WORDS_FOR_PREDICTION, 1)
                y_batch_encoded = y_batch_encoded.reshape(last_batch, len(vocab_dict))
            else:
                x_batch = x_batch.reshape(batch_size, NUM_WORDS_FOR_PREDICTION, 1)
                y_batch_encoded = y_batch_encoded.reshape(batch_size, len(vocab_dict))

            _, acc, loss, prediction = sess.run([train_step, accuracy, loss_op, y_output],
                                                feed_dict={x_input: x_batch, y_input: y_batch_encoded})
            print("Step: " + str(i) + ", loss: {:.4f}".format(loss) + ", accuracy: {:.2f}".format(acc * 100))
            print("Prediction: ", interpret_results(prediction))

    saver.save(sess, './text_predictor.ckpt')
