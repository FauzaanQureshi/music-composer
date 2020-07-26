import numpy as np
import pandas as pd


def processMidi(inputCSV):
    """
        Returns tuple containing Input and Target values to be passed in tensor dataset
    """
              
    df = pd.read_csv(inputCSV, # Read Midi data
                      header=None, names=['Channel', 'Time', 'Type', 'C4', 'C5', 'C6']).fillna(0)        # Fill Missing Values with 0
    
    
    df = df[df['Type']!=' Header']
    df = df[df['Type']!=' Start_track']
    df = df[df['Type']!=' Title_t']
    df = df[df['Type']!=' Copyright_t']
    df = df[df['Type']!=' Text_t']
    df = df[df['Type']!=' Time_signature']
    df = df[df['Type']!=' Key_signature']
    df = df[df['Type']!=' Marker_t']
    df = df[df['Type']!=' End_track']
    df = df[df['Type']!=' End_of_file']
    
    df = df.drop('Channel', 1)  #Drop Channel as it remains same(1) everywhere
    df['Time'] = df['Time'].astype('uint32')
    df['Type'] = df['Type'].astype(pd.api.types.CategoricalDtype(categories=[' Tempo', ' Note_on_c', ' Program_c', ' Control_c'],ordered=True))
    df['C4'] = df['C4'].astype('int')
    df['C5'] = df['C5'].astype('int')
    df['C6'] = df['C6'].astype('int')
    
    df['Type'] = df.Type.cat.codes
    df = df.reset_index(drop=True)
    #df.to_csv('Generated.csv',index=False)
    #print(df.iloc[:-1]) # Input:    Dataframe with last row removed
    #print(df.iloc[1:])  # Target:   Dataframe with first row removed
    #processMidi.shape = asarray([df.iloc[1:].values]).shape
    processMidi.shape = df.iloc[1:].shape
    #print(processMidi.shape)
    return [df.iloc[:-1].values, df.iloc[1:].values]
    
    
def makeMidi(output_csv):
    """
        Makes csv file that can be passed to CscMidi.exe.
        
        :param output_csv: Location of csv file containing predictions from the NN
    """
    df = pd.read_csv(output_csv)#, header=None, names=['Channel', 'Time', 'Type', 'C4', 'C5', 'C6'])
    
    end_index = df.shape[0]
    
    midi_header = pd.DataFrame({'Channel':'0', 'Time':' 0', 'Type':' Header', 'C4':' 0', 'C5':' 1', 'C6':' 480'}, index=[-2])
    midi_start_track = pd.DataFrame({'Channel':'1', 'Time':' 0', 'Type':' Start_track'}, index=[-1])
    midi_end_track = pd.DataFrame({'Channel':'1', 'Time':df['Time'].values[-1], 'Type':' End_track'}, index=[end_index])
    midi_eof  = pd.DataFrame({'Channel':'0', 'Time':'0', 'Type':' End_of_file'}, index=[end_index+1])
    
    Type_categories = [' Tempo', ' Note_on_c', ' Program_c', ' Control_c']
    Type_decode = {key: value for key, value in enumerate(Type_categories)}
    
    df.insert(0, 'Channel', 1)
    df.loc[df['Type']==0, df.columns[4:]] = ''
    df.loc[df['Type']==2, df.columns[5:]] = ''
    
    df['Type'] = df['Type'].astype('category')
    df['Type'] = df.Type.cat.rename_categories(Type_decode)
    
    df = df.append(midi_header, ignore_index=False)
    df = df.append(midi_start_track, ignore_index=False)
    df = df.append(midi_end_track, ignore_index=False)
    df = df.append(midi_eof, ignore_index=False)
    df = df.sort_index().reset_index(drop=True)
    df = df[['Channel', 'Time', 'Type', 'C4', 'C5', 'C6']]
    
    df['C5'] = df['C5'].astype('Int64', errors='ignore')
    df['C6'] = df['C6'].astype('Int64', errors='ignore')
    
    df.to_csv('Out_'+output_csv, index=False, header=False)
    #print(df)


if __name__ == '__main__':
    #pass
    #makeMidi('Generated.csv')
    processMidi('E:\\tmp\\midicsv-1.1\\s.csv')
    #print(np.asanyarray([[1,2,3,4,5]]).reshape(1,5,1), end='\n\n\n')
    #print(np.asanyarray([[1,2,3,4,5]]))

