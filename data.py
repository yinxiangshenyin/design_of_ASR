# the train_data and the test_data of ASR_training
#this data is for chinese

import error
import os
import numpy as np
import python_speech_features
#import librosa
import pickle
from collections import Counter
pointer=0
class Train_data:
    train_wav_file_path = r'D:\workplace\Python\SpeechRecognition\speech_sample\wav\wav\train'
    train_label_file_path = r'D:\workplace\Python\SpeechRecognition\speech_sample\doc\doc\trans\train.word.txt'
    train_mfcc=[]
    train_label=[]
    encode_dict={}
    wav_max_len=0
    label_max_len=0
    label_class_num=0
    decode_dict={}


    save_mfcc=r'..\data for ASR\save_mfcc.txt'
    #save_mfcc =r'test.txt'
    save_labels=r'..\data for ASR\save_labels.txt'
    save_word_class_dict=r'..\data for ASR\save_word_class_dict.txt'


    #train_data
    # function get all the wav_file in the path
    def Get_wav_file(self,path=""):
        wave_file_list=[]
        if  path=="":
            return wave_file_list
        else:
            for(dirpath,dirname,filenames) in os.walk(path):
                for filename in filenames:
                    if filename.endswith('.wav') or filename.endswith('.WAV'):
                        file_path=os.sep.join([dirpath,filename])
                        if os.stat(file_path).st_size<240000:
                            continue
                        wave_file_list.append(file_path)
            return wave_file_list

    def Get_wav_label(self,path=""):
        label_dic = {}
        if path=="":
            return label_dic
        else:
            with open(path,'r',encoding='utf-8') as f:
                for label in f:
                    label=label.strip('\n')
                    label_id=label.split(' ',1)[0]
                    label_content=label.split(' ',1)[1]
                    label_content=label_content.replace(' ','')
                    label_dic[label_id]=label_content
                return label_dic

    def Get_original_train_data(self):
        wav_data=self.Get_wav_file(self.train_wav_file_path)
        label_data=self.Get_wav_label(self.train_label_file_path)
        original_wav=[]
        original_label=[]
        for wav_file in wav_data:
            wav_id=os.path.basename(wav_file).split('.')[0]
            if wav_id in label_data:
                original_wav.append(wav_file)
                original_label.append(label_data[wav_id])
        return original_wav,original_label

    def Get_max_len_from_list(self,list):
        return np.max([len(item) for item in list])



    def Get_label_class(self,labels):
        all_word=[]
        for label in labels:
            all_word+=[word for word in label]
        counter=Counter(all_word)
        counter_pair=sorted(counter.items(),key=lambda x:-x[1])
        word_class,_=zip(*counter_pair)
        word_class_dic=dict(zip(word_class,range(len(word_class))))
        return word_class_dic

    def Savedata(self,filemane,data):
        f = open(filemane, 'wb')
        pickle.dump(data, f, 0)
        f.close()

    def Loaddata(self,filename):
        f=open(filename,'rb')
        data=pickle.load(f)
        return data


    def Data_process(self):
        if os.path.exists(self.save_labels) and os.path.exists(self.save_mfcc) and os.path.exists(self.save_word_class_dict):
            print("Loading data from txt")
            self.label_class_dict=self.Loaddata(self.save_word_class_dict)
            self.train_mfcc=self.Loaddata(self.save_mfcc)
            self.train_label=self.Loaddata(self.save_labels)

            #self.train_mfcc = np.array(self.Loaddata(self.save_mfcc)).reshape((len(self.train_mfcc),-1,20))
            print("Loading sucessful")
        else:
            original_wav,original_label=self.Get_original_train_data()
            self.encode_dict=self.Get_label_class(original_label)
            self.decode_dict={v:k for k,v in self.encode_dict.items()}

            label_to_num=lambda word:self.label_class_dict.get(word,0)
            self.train_label=[list(map(label_to_num,label)) for label in original_label]
            #self.train_mfcc=self.Get_MFCC(original_wav)
            self.Savedata(self.save_mfcc,self.train_mfcc)
            self.Savedata(self.save_word_class_dict ,self.label_class_dict)
            self.Savedata(self.save_labels,self.train_label)

        self.wav_max_len=self.Get_max_len_from_list(self.train_mfcc)
        self.label_max_len=self.Get_max_len_from_list(self.train_label)
        self.pointer=0
        for item in self.train_mfcc:
            while len(item)<self.wav_max_len:
                item.append([0]*20)
        for item in self.train_label:
            while len(item)<self.label_max_len:
                item.append(-1)

    def Get_next_batch(self,batch_size):

        batch_wav=np.array(self.train_mfcc)[self.pointer:self.pointer+batch_size]
        batch_label=np.array(self.train_label)[self.pointer:self.pointer+batch_size,:]
        self.pointer=self.pointer+batch_size
        return batch_wav,batch_label


    def decod(self,input):
        str=[]
        for item in input.eval():
            str.append(self.decode_dict[item])
        return str
        label_to_num=lambda word:self.label_class_dict.get(word,0)
        self.train_label=[list(map(label_to_num,label)) for label in original_label]







