'''
Created on Aug 13, 2018

@author: Inthuch Therdchanakul
'''
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

class LSTMNetwork():
    def __init__(self, n_layer, lstm_unit, input_shape, labels, base_dir, feature='VGG16', output_activation='softmax'):
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
        self.base_dir = base_dir + 'LSTM/' + feature + '/'
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
        
#     def evaluate(self, X_val, y_val):
#         # evaluate_vgg16 the model with validation set
#         model = load_model(self.model_file)
#         scores = model.evaluate(X_val, y_val)
#         print('val_loss: {}, val_acc: {}'.format(scores[0], scores[1]))
#         
#         y_true, y_pred = get_predictions_and_labels(model, X_val, y_val)
#         cm = confusion_matrix(y_true, y_pred)
#         cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         df = pd.DataFrame(cm_percent, index=self.labels, columns=self.labels)
#         df.index.name = 'Actual'
#         df.columns.name = 'Predicted'
#         df.to_csv(self.base_dir+self.model_dir+'cm_val.csv', float_format='%.4f')
#         
#         # plot percentage confusion matrix
#         fig1, ax1 = plt.subplots()
#         plot_confusion_matrix(cm_percent, class_names=[i for i in range(1, len(self.labels) + 1)])
#         plt.savefig(self.base_dir + self.model_dir + 'cm_percent_val.png', format='png')
#         # plot normal confusion matrix
#         fig2, ax2 = plt.subplots()
#         plot_confusion_matrix(cm, float_display='.0f', class_names=[i for i in range(1, len(self.labels) + 1)])
#         plt.savefig(self.base_dir + self.model_dir + 'cm_val.png', format='png')
#         
#         plt.show()
        
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