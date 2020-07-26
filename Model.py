from DataProcessing import processMidi

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.losses import mean_squared_error, sparse_categorical_crossentropy

import numpy as np


def createMultiOutModel(data_dim):
    inputs = Input((data_dim[1], 1))
    model = GRU(15)(inputs)
    model = Dropout(0.375)(model)

    out1 = Dense(1, 'relu')(model)
    out2 = Dense(1, 'relu')(model)
    out3 = Dense(1, 'relu')(model)
    out4 = Dense(1, 'relu')(model)
    out5 = Dense(1, 'relu')(model)

    model = Model(inputs, [out1, out2, out3, out4, out5])
    model.compile(optimizer=Adadelta(learning_rate=0.25),
                  loss=[mean_squared_error,
                        mean_squared_error,
                        mean_squared_error,
                        sparse_categorical_crossentropy,
                        mean_squared_error],
                  metrics=['accuracy'])
    return model


def createModel(data_dim):
    model = Sequential()
    print(data_dim)
    model.add(GRU(15, input_shape=(data_dim[1], 1)))
    model.add(Dropout(0.375))
    # model.add(GRU(7))
    # model.add(Dropout(0.25))
    model.add(Dense(5, activation='relu'))
    model.summary()
    model.compile(optimizer=Adadelta(),
                  loss=[#"mean_squared_error",
                        #'sparse_categorical_crossenropy',
                        #"mean_squared_error",
                        #"mean_squared_error",
                        "mean_squared_error"],
                  metrics=['accuracy'])
    return model


def trainModel(inputCSV, batch_size=8, epochs=4):
    # dataset = Dataset.from_tensor_slices(processMidi(inputCSV))
    data = processMidi(inputCSV)
    #print(np.array([np.array(i).reshape(-1, 1) for i in data[1][0:batch_size]]).shape)
    #exit(0)
    #np.random.shuffle(data)

    # model = createModel(processMidi.shape)
    model = createMultiOutModel(processMidi.shape)
    try: model.load_weights('comp.h5')
    except OSError: pass
    # dataset.batch(64)
    # model.fit(dataset, epochs=1, batch_size=64)

    # USE WITH createModel()
    # model.fit(np.asarray([data[0]]).reshape(-1, 5, 1),
    #           np.asarray([data[1]]).reshape(-1, 5, 1),
    #           epochs=epochs,
    #           batch_size=batch_size)

    # USE WITH createMultiOutModel()
    y = np.transpose(data[1])
    model.fit(np.asarray([data[0]]).reshape(-1, 5, 1),
              [y[0], y[1], y[2], y[3], y[4]],
              epochs=epochs,
              batch_size=batch_size)
    model.save_weights('comp.h5')


def testModel():
    # model = createModel((8191, 5))
    model = createMultiOutModel((8191, 5))
    model.load_weights('comp.h5')
    model_input = [156, 3, 0, 64, 127]  # [0,0,325007,0,0]
    n_lines = 5
    # csv_out = open('output.csv')
    for i in range(n_lines):
        model_input = np.array(model.predict(np.asanyarray([model_input]).reshape(1, 5, 1))).reshape(-1, 5)
        print(np.array(model_input).reshape(-1, 5))


if __name__ == '__main__':
    trainModel('ReGenerated.csv', batch_size=64, epochs=32) if input('Train? <"y"/n> ')=='y' else testModel()
