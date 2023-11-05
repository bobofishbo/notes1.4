import tensorflow as tf
from tensorflow import keras
import gspread
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from oauth2client.service_account import ServiceAccountCredentials


from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os

import string
import secrets
import hashlib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def sha256(message):
  m = hashlib.sha256()
  message = message.encode()
  message = bytes(message)
  m.update(message)
  return m.hexdigest()

def generatePassword(length):
  alphabet = string.ascii_letters + string.digits + string.punctuation
  while True:
      password = ''.join(secrets.choice(alphabet) for i in range(length))
      if (any(c.islower() for c in password)
              and any(c.isupper() for c in password)
              and sum(c.isdigit() for c in password) >= 3
              and sum(c.isdigit() for c in password) <= 5
              and sum(c.isalpha() for c in password) <= 5):
          break
  return password

def generateUsername(length):
  alphabet = string.ascii_letters + string.digits
  while True:
    username = ''.join(secrets.choice(alphabet) for i in range(length))
    if (any(c.islower() for c in username)
            and any(c.isupper() for c in username)
            and sum(c.isdigit() for c in username) >= 3):
        break
  return username


def getDatabase(databaseLength, passwordLength, worksheet):
  sa = gspread.service_account(filename='client_key.json')
  sh = sa.open('sha256Dataset')
  wks = sh.worksheet(worksheet)
  for i in range(2, databaseLength + 2):
    password = generatePassword(passwordLength)
    wks.update(f'A{i}', password)
    wks.update(f'B{i}', sha256(password))
  google_sheet_key = '1RGOt8g_iLHCghjiDVwZFmFS2uZbtf2NH5kqy8hRh468'
  sheet_name = worksheet
  csv_file_path = 'output.csv'

  scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
  credentials = ServiceAccountCredentials.from_json_keyfile_name('client_key.json', scope)
  client = gspread.authorize(credentials)

  sheet = client.open_by_key(google_sheet_key).worksheet(sheet_name)

  data = sheet.get_all_values()

  df = pd.DataFrame(data[1:], columns=data[0])

  df.to_csv(csv_file_path, index=False)
  print(df)
  return df


def runModel(databaseLength, passwordLength, worksheet):

  df = getDatabase(databaseLength, passwordLength, worksheet)

  x = df['Password'].values
  y = df['Password Encryption'].values

  label_encoder = LabelEncoder()
  y_encoded = label_encoder.fit_transform(y)

  X_train, X_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

  tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, filters = None, lower = False)
  tokenizer.fit_on_texts(X_train)

  X_train_seq = tokenizer.texts_to_sequences(X_train)
  X_test_seq = tokenizer.texts_to_sequences(X_test)

  max_length = max(len(seq) for seq in X_train_seq + X_test_seq)
  X_train_padded = tf.keras.preprocessing.sequence.pad_sequences(X_train_seq, maxlen = max_length)
  X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test_seq, maxlen = max_length)

  input_layer = Input(shape=(X_train_padded.shape[1], ))
  embedding_layer = Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=64)(input_layer)
  lstm_layer = LSTM(64)(embedding_layer)
  output_layer = Dense(len(label_encoder.classes_), activation='softmax')(lstm_layer)

  model = Model(inputs=input_layer, outputs=output_layer)

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  model.summary()

  model.fit(X_train_padded, y_train, epochs=60, batch_size=32, validation_data=(X_test_padded, y_test))

  print('Done!')

runModel(30, 8, 'Dataset10')