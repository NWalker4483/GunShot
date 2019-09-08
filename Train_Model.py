from keras.callbacks import ModelCheckpoint
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, LSTM, Conv1D
from keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from keras.optimizers import Adam

from LoadDataHelpers import LoadData

#TODO: Convert to format below 
"""
model = Sequential()
model.add(Dense(32, input_dim=784))
model.add(Activation('relu'))
""" 
def BuildModel(input_shape):
    """
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape = input_shape)
    
    # Step 1: CONV layer
    X = Conv1D(196, kernel_size=15, strides=4)(X_input)   # CONV1D
    X = BatchNormalization()(X)                           # Batch normalization
    X = Activation('relu')(X)                             # ReLu activation
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)

    # Step 2: First GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)      # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    X = BatchNormalization()(X)                           # Batch normalization
    
    # Step 3: Second GRU Layer
    X = GRU(units = 128, return_sequences = True)(X)      # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    X = BatchNormalization()(X)                           # Batch normalization
    X = Dropout(0.8)(X)                                   # dropout (use 0.8)
    
    # Step 4: Time-distributed dense layer
    X = TimeDistributed(Dense(1, activation = "sigmoid"))(X) # time distributed  (sigmoid)
    model = Model(inputs = X_input, outputs = X)    
    return model  

SPQ = 1927 #5511 # The number of samples feed from the spectogram -> model 
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram

model = BuildModel(input_shape = (SPQ, n_freq))
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.summary()

X, Y = LoadData()

model.fit(X, Y, batch_size = 1, epochs=1)