from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

def write_text(sentences, output_file):
    """Write text from sentences."""
    with open(output_file, "w+") as out_stream:
        for sentence in sentences:
            out_stream.write(sentence + "\n")
            
def read_file(file_name: str):
    """Read file and returns paragraphs."""
    output = []
    with open(file_name, "r") as stream:
        for line in stream:
            tokens = line.strip()
            output.append(tokens)
    return output

def compute_bleu(pred_file_path: str, target_file_path: str):
    """Compute bleu score.

    Args:
        pred_file_path: the file path that contains the predictions.
        target_file_path: the file path that contains the targets (also called references).

    Returns: Bleu score

    """
    out = subprocess.run(
        [
            "sacrebleu",
            "--input",
            pred_file_path,
            target_file_path,
            "--tokenize",
            "none",
            "--sentence-level",
            "--score-only",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    lines = out.stdout.split("\n")
    scores = [float(x) for x in lines[:-1]]
    return sum(scores) / len(scores)

# Converts the unicode file to ascii
def unicode_to_ascii(s):
  return ''.join(c for c in unicodedata.normalize('NFD', s)
      if unicodedata.category(c) != 'Mn')



def preprocess_sentence(w):
  w = unicode_to_ascii(w.lower().strip())

  # creating a space between a word and the punctuation following it
  # eg: "he is a boy." => "he is a boy ."
  # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
  w = re.sub(r"([?.!,¿])", r" \1 ", w)
  w = re.sub(r'[" "]+', " ", w)

  # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
  w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

  w = w.rstrip().strip()

  # adding a start and an end token to the sentence
  # so that the model know when to start and stop predicting.
  w = '<start> ' + w + ' <end>'
  return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, SPANISH]
def create_dataset(lines, num_examples):
  #lines = io.open(path, encoding='UTF-8').read().strip().split('\n')

  word_pairs = [[preprocess_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]

  return zip(*word_pairs)

def max_length(tensor):
  return max(len(t) for t in tensor)

def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer

def tokenize_with_tokenizer(lang, tokenizer):
    tensor = tokenizer.texts_to_sequences(lang)

    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor

def load_dataset(i, t, num_examples=None):
  # creating cleaned input, output pairs
  #targ_lang, inp_lang = create_dataset(path, num_examples)

  input_tensor, inp_lang_tokenizer = tokenize(i)
  target_tensor, targ_lang_tokenizer = tokenize(t)

  return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

def loss_function(real, pred, loss_object):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden, encoder, decoder, targ_lang, BATCH_SIZE,loss_object, optimizer):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    prediction = []
    target = []
    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
      prediction.append(predictions)
      target.append(targ[:, t])

      loss += loss_function(targ[:, t], predictions, loss_object)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss, target, prediction

def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot

# function for plotting the attention weights
def plot_attention(attention, sentence, predicted_sentence):
  fig = plt.figure(figsize=(10,10))
  ax = fig.add_subplot(1, 1, 1)
  ax.matshow(attention, cmap='viridis')

  fontdict = {'fontsize': 14}

  ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
  ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
  ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

  plt.show()


def main():
    train_lines = read_file('test.train')
    #val_lines = read_file('test.val')

    en_train, fr_train = create_dataset(train_lines, None)
    #en_val, fr_val = create_dataset(val_lines, None)

    input_tensor_train, target_tensor_train, inp_lang_tokenizer, targ_lang_tokenizer = load_dataset(en_train, fr_train)
    #input_tensor_val = tokenize_with_tokenizer(en_val, inp_lang_tokenizer)
    #target_tensor_val = tokenize_with_tokenizer(fr_val, targ_lang_tokenizer)

    max_length_targ, max_length_inp = max_length(input_tensor_train), max_length(target_tensor_train)

    BUFFER_SIZE = len(input_tensor_train)
    BATCH_SIZE = 4
    steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
    embedding_dim = 256
    units = 1024
    vocab_inp_size = len(inp_lang_tokenizer.word_index)+1
    vocab_tar_size = len(targ_lang_tokenizer.word_index)+1

    dataset_train = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train))
    dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)

    #dataset_val = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val))
    #dataset_val = dataset_val.batch(BATCH_SIZE, drop_remainder=True)

    example_input_batch, example_target_batch = next(iter(dataset_train))
    example_input_batch.shape, example_target_batch.shape

    encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
    sample_hidden = encoder.initialize_hidden_state()
    sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

    attention_layer = BahdanauAttention(10)
    attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

    decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                                  sample_hidden, sample_output)

    optimizer = tf.keras.optimizers.Adam()
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')


    EPOCHS = 10

    for epoch in range(EPOCHS):
      start = time.time()

      enc_hidden = encoder.initialize_hidden_state()
      total_loss = 0

      pred = []
      tar = [] 
      for (batch, (inp, targ)) in enumerate(dataset_train.take(steps_per_epoch)):
        batch_loss, target, prediction = train_step(inp, targ, enc_hidden, encoder,decoder,  targ_lang_tokenizer, BATCH_SIZE, loss_object, optimizer)
        total_loss += batch_loss

        pred += prediction
        tar += target
        if batch % 100 == 0:
          print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                       batch,
                                                       batch_loss.numpy()))
      write_text(pred, 'Predictionnns')
      write_text(tar, 'Targetttss')
      bleu = compute_bleu('Predictionnns', 'Targetttss')

      print(f'Epoch {epoch+1} Bleu {bleu}')
      print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                          total_loss / steps_per_epoch))
      print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))


if __name__ == '__main__':
    main()
