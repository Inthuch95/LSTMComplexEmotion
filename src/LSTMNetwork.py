'''
Created on Aug 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
import time
import os

class LSTMNetwork():
    def __init__(self, n_layer, lstm_unit, input_shape, labels, base_dir, feature='vgg16', output_activation='softmax'):
        self.labels = labels
        
        self.model = Sequential()
        if n_layer > 1:
            self.model.add(LSTM(lstm_unit, return_sequences=True, input_shape=input_shape,
                       dropout=0.2))
            layer_count = 1
            while layer_count < n_layer:
                if layer_count == n_layer-1:
                    self.model.add(LSTM(lstm_unit, return_sequences=False, dropout=0.2))
                else:
                    self.model.add(LSTM(lstm_unit, return_sequences=True, dropout=0.2))
                layer_count += 1
        else:
            self.model.add(LSTM(lstm_unit, return_sequences=False, input_shape=input_shape,
                       dropout=0.2))
        nb_class = len(self.labels)
        self.model.add(Dense(nb_class, activation=output_activation))
        
        current_time = time.strftime("%Y%m%d-%H%M%S")
        self.base_dir = base_dir + 'LSTM/' + feature.upper() + '/'
        self.model_dir = 'LSTM_' + str(n_layer) + '_' + str(lstm_unit) + '_' + current_time + '/'
        filename = 'LSTM.h5'
        self.model_file = self.base_dir + self.model_dir + filename
    
    def train(self, X_train, y_train, X_val, y_val, epochs=250, batch_size=32, 
              optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']):
        # compile and train the model
        if not os.path.exists(self.base_dir + self.model_dir):
            os.mkdir(self.base_dir + self.model_dir)
        log_dir = self.base_dir + self.model_dir + 'log/'
        os.mkdir(log_dir)
        self.model.compile(optimizer=optimizer,
                      loss=loss, 
                      metrics=metrics)
        callbacks = [ModelCheckpoint(self.model_file, monitor='val_loss', save_best_only=True, verbose=0),
                     TensorBoard(log_dir=log_dir, write_graph=True)]
        self.model.fit(X_train, y_train,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(X_val, y_val),
                callbacks=callbacks)
        
    def load_best_model(self):
        model = load_model(self.model_file)
        return model
        
    def compare_model(self, X_val, y_val):
        folder_list = [model_dir for model_dir in os.listdir(self.base_dir) if 'LSTM' in model_dir]
        for folder in folder_list:
            filename = 'LSTM.h5'
            path = os.path.join(self.base_dir, folder, filename)
            model = load_model(path)
            scores = model.evaluate(X_val, y_val)
            print('model: {}, val_loss: {}, val_acc: {}'.format(folder, scores[0], scores[1]))

if __name__ == '__main__':
    pass