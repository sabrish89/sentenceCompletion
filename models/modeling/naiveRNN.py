from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Bidirectional, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_accuracy, categorical_crossentropy
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def naiveRNN(vocab_size, seqLength, learning_rate=0.001):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, activation="relu"), input_shape=(seqLength, vocab_size)))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    optimizer = Adam(lr=learning_rate)
    model.compile(loss=categorical_crossentropy, optimizer=optimizer, metrics=[categorical_accuracy])
    return model

def trainModel(md, X, y):
    early_stp = EarlyStopping(monitor='val_loss', verbose=1,patience=8, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath="../models/data/simpleLSTM.hdf5", \
                    monitor='val_loss', verbose=1, mode='auto', period=2)
    callbacks = [checkpoint, early_stp]
    history = md.fit(X, y,
                     batch_size=2000,
                     shuffle=True,
                     epochs=50,
                     callbacks=callbacks,
                     validation_split=0.1)
    plotHistory(history)
    return md

def plotHistory(history):
    '''
    Plot keras history for model
    '''
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()