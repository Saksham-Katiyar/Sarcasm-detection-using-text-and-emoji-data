import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D, SimpleRNN
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import re


data = pd.read_csv('./Data/Dataset.csv')
print(data.columns)
data = data[['Text','Sarcasm']]


data['Text'] = data['Text'].astype(str)

data['Text'] = data['Text'].apply(lambda x: x.lower())
data['Text'] = data['Text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]','',x)))
# Function to add space after every alphanumeric sequence

# def add_space_after_alphanumeric(text):
#     return re.sub(r'(\w+)', r'\1 ', text)

# # Apply the function to the 'Comments' column
# data['Comments'] = data['Comments'].apply(add_space_after_alphanumeric)

# data['Emojis'] = data['Emojis'].astype(str)

print(data[ data['Sarcasm'] == 1].size)
print(data[ data['Sarcasm'] == 0].size)

# for idx,row in data.iterrows():
#     row[0] = row[0].replace('rt',' ')

data.head()


max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=' .,?/-!#&')
tokenizer.fit_on_texts(data['Text'].values)
X = tokenizer.texts_to_sequences(data['Text'].values)
X = pad_sequences(X)

Y = pd.get_dummies(data['Sarcasm']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.33, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

validation_size = 2000
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
gru_model = load_model('/home/namitha/Downloads/GRU_only_text_upadted.h5')
lstm_model = load_model('/home/namitha/Downloads/LSTM_only_text_updated.h5')
rnn_model = load_model('/home/namitha/Downloads/RNN_only_text_updated.h5')
bilstm_model = load_model('/home/namitha/Downloads/BiLSTM_only_text_updated.keras')



pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0

for x in range(len(X_test)):
    
    result_rnn = rnn_model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose=0)[0]
    result_lstm = lstm_model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose=0)[0]
    result_bilstm = bilstm_model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose=0)[0]

    #result_gru =  gru_model.predict(X_test[x].reshape(1,X_test.shape[1]),batch_size=1,verbose=0)[0]

    final_pred = (result_rnn + result_lstm + result_bilstm)/3


    if np.argmax(final_pred) == np.argmax(Y_test[x]):
        if np.argmax(Y_test[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1
       
    if np.argmax(Y_test[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos_acc", pos_correct/pos_cnt*100, "%")
print("neg_acc", neg_correct/neg_cnt*100, "%")









