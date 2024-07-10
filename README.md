# chatbot-using-TensorFlow

-------------------------------------------------this code has failed------------------------------------------------------------------------------

-----------------------------------------sorry for that--------------------------------------------------
----------------------------------too many bugs ---------------------------------------------------------------
import tensorflow as tf
import numpy as np
import nltk
import re
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')

# Sample data (replace this with your actual data loading mechanism)
input_texts = [
    "Hi, how are you?",
    "What's your name?",
    "Where do you live?",
    "Tell me something interesting."
]
target_texts = [
    "I am fine, thank you!",
    "I am a chatbot.",
    "I live in the digital world.",
    "Sure! Did you know that honey never spoils?"
]

# Add start and end tokens to target texts
target_texts = ['<start> ' + text + ' <end>' for text in target_texts]

# Data Preprocessing
def preprocess_sentence(sentence):
    sentence = sentence.lower()
    sentence = re.sub(r"([?.!,])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,]+", " ", sentence)
    sentence = sentence.strip()
    return sentence

def tokenize(sentences):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
    tokenizer.fit_on_texts(sentences)
    tokenizer.word_index['<pad>'] = 0
    tensor = tokenizer.texts_to_sequences(sentences)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer

# Preprocess and tokenize sentences
preprocessed_input_texts = [preprocess_sentence(sentence) for sentence in input_texts]
preprocessed_target_texts = [preprocess_sentence(sentence) for sentence in target_texts]

input_tensor, input_tokenizer = tokenize(preprocessed_input_texts)
target_tensor, target_tokenizer = tokenize(preprocessed_target_texts)

# Ensure <start> and <end> tokens are included
if '<start>' not in target_tokenizer.word_index:
    target_tokenizer.word_index['<start>'] = len(target_tokenizer.word_index) + 1
if '<end>' not in target_tokenizer.word_index:
    target_tokenizer.word_index['<end>'] = len(target_tokenizer.word_index) + 1

# Define constants
vocab_size_input = len(input_tokenizer.word_index) + 1
vocab_size_target = len(target_tokenizer.word_index) + 1
embedding_dim = 256
units = 1024
batch_size = 2
max_length_inp = max(len(t) for t in input_tensor)
max_length_targ = max(len(t) for t in target_tensor)

# Create TensorFlow dataset
dataset = tf.data.Dataset.from_tensor_slices((input_tensor, target_tensor)).shuffle(len(input_tensor))
dataset = dataset.batch(batch_size, drop_remainder=True)
steps_per_epoch = len(input_tensor) // batch_size

# Define the Encoder
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
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# Define the Decoder
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
        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, x, hidden, enc_output):
        x = self.embedding(x)
        context_vector, _ = self.attention([hidden, enc_output])
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state

# Initialize encoder and decoder
encoder = Encoder(vocab_size_input, embedding_dim, units, batch_size)
decoder = Decoder(vocab_size_target, embedding_dim, units, batch_size)

# Training Process
optimizer = tf.keras.optimizers.Adam()

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']] * batch_size, 1)

        for t in range(1, targ.shape[1]):
            predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_output)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

# Training Loop
EPOCHS = 10

for epoch in range(EPOCHS):
    enc_hidden = encoder.initialize_hidden_state()
    total_loss = 0

    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        print(f"Batch {batch+1}, Input: {inp}, Target: {targ}")

        if '<start>' not in target_tokenizer.word_index:
            raise KeyError("'<start>' token not found in target tokenizer word index")

        batch_loss = train_step(inp, targ, enc_hidden)
        total_loss += batch_loss

    print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')

# Inference
def evaluate(sentence):
    sentence = preprocess_sentence(sentence)
    inputs = [input_tokenizer.word_index.get(i, input_tokenizer.word_index['<unk>']) for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs], maxlen=max_length_inp, padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([target_tokenizer.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden = decoder(dec_input, dec_hidden, enc_out)
        predicted_id = tf.argmax(predictions[0]).numpy()
        result += target_tokenizer.index_word.get(predicted_id, '') + ' '

        if target_tokenizer.index_word.get(predicted_id, '') == '<end>':
            return result.strip()

        dec_input = tf.expand_dims([predicted_id], 0)

    return result.strip()

def respond(input_text):
    response = evaluate(input_text)
    print(f'Chatbot: {response}')

# Interaction
while True:
    input_text = input('You: ')
    if input_text.lower() == 'quit':
        break
    respond(input_text)


--------------------------------------code update #3---------------------------------------------------------------------------



--------------------------------not going to use ChatGPT it will hurt my ego-------------------------------------
