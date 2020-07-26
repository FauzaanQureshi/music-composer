import numpy as np
from keras.models import Model
from keras.layers import GRU, Dense, Dropout, Input
from matplotlib.pyplot import *
from Scaler import *

def createNpModel():
    inputs = Input((1, 1))
    model = GRU(512, return_sequences=True, reset_after=True)(inputs)
    model = Dropout(0.25)(model)
    model = GRU(512,reset_after=True)(model)
    model = Dropout(0.25)(model)
    model = Dense(1, activation='relu')(model)
    model = Model(inputs, model)
    model.compile(optimizer='adam', loss='mae',metrics=['accuracy'])
    return model
    
    
def trainNpModel(data):
    batches = data.shape[0]//256
    
    model = createNpModel()
    #try: model.load_weights('NpModelWeights.h5')
    #except OSError: pass
    model.summary()
    if input('Exit? ')=='y': exit(0)
    for i in range(batches//2):
        model.fit(data[256*i:256*(i+1)], data[1+256*i:1+256*(i+1)].reshape(-1, 1), batch_size=256, epochs=4, shuffle=False)
    
    model.save_weights('NpModelWeights.h5')
    

def testNpModel():
    model = createNpModel()
    model.load_weights('NpModelWeights.h5')
    seq = np.array([sc.scale(data[0])])
    t = [sc.scale(data[0])]
    seq_size = 2560 #data.shape[0]//2
    print('Target size : ', seq_size)
    print('Current size: ', seq.shape[0])
    while len(t)<seq_size:
        t.append(model.predict(np.array(t[-1]).reshape(-1, 1, 1))[0])
        if len(t)%256==0:
            print(len(t)//256)
    seq = np.append(seq, t)
    #print(seq)
    return sc.descale(seq)

    
def plotNpModel(overlap_plot = False):
    if overlap_plot:
        figure(figsize=(20, 10))
        plot(range(output.shape[0]), data[:output.shape[0]])
        plot(range(output.shape[0]), output)
    
    else:
        figure(1, figsize=(20, 10))
        title('Original')
        plot(range(output.shape[0]), data[:output.shape[0]])

        figure(2, figsize=(20, 10))
        title('Generated')
        plot(range(output.shape[0]), output)

    show()
   
   
def scrib(data=None, scaler=None, output=None):
    if not output:
        output = np.load('pridicted.npy', allow_pickle=True)
    figure(figsize=(20, 10))
    plot(range(output.shape[0]), scaler.scale(data[:output.shape[0]]))
    plot(range(output.shape[0]), scaler.scale(output))
    show()
    
if __name__=='__main__':
    data = np.load('FurElise.npy')
    sc = Scaler(data)
    
    ch = int(input('\n\n1. Train\n2.Test\n3.Plot\n\n0. Exit\n\nChoice:'))
    
    while ch:
        if ch==1:
            trainNpModel(sc.scale(data).reshape(-1, 1, 1))
    
        elif ch==2:
            output = testNpModel()
            np.save('pridicted', output, fix_imports=False)
            #sc.scale(data[0])
        
        elif ch==3:
            plotNpModel(overlap_plot = True)
            
        elif ch==9:
            scrib(data=data, scaler=sc)
        
        else:
            print('\n\nINVALID CHOICE!!!\n\n')
        
        ch = int(input('\n\n1. Train\n2.Test\n3.Plot\n\n0. Exit\n\nChoice:'))
