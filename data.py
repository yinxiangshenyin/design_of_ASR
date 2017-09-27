# the train_data and the test_data of ASR_training
#this data is for chinese

import error
import os
import numpy as np
import python_speech_features
#import librosa
import threading
from queue import Queue
import pickle
import csv
import time
from collections import Counter
class Train_data:
    train_wav_file_path = r'D:\workplace\Python\SpeechRecognition\speech_sample\wav\wav\train'
    train_label_file_path = r'D:\workplace\Python\SpeechRecognition\speech_sample\doc\doc\trans\train_word.txt'
    train_mfcc=[]
    train_label=[]
    encode_dict={}
    wav_max_len=8911
    label_max_len=8911
    label_class_num=0
    decode_dict={}
    label_queue = Queue()
    mfcc_queue = Queue()
    label_lock = threading.Lock()
    mfcc_lock = threading.Lock()
    read_label_threading=None
    read_mfcc_threading = None

    save_mfcc=r'..\data for ASR\save_mfcc.csv'
    save_labels=r'..\data for ASR\save_labels.csv'
    save_word_class_dict=r'..\data for ASR\save_word_class_dict.txt'

    test_mfcc=r'..\data for ASR\test_mfcc.csv'
    test_labels = r'..\data for ASR\test_labels.csv'


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
    def Savedata_csv(self,filemane,data):
        csvfile = open(filemane, 'w', newline="")
        writer = csv.writer(csvfile)
        m = len(data)
        print(m)
        for i in range(m):
            writer.writerow(data[i])
        csvfile.close()

    def readcsvdata_mfcc(self, filename, mfcc_queue, mfcc_lock):
        x = []
        with open(filename, 'r') as csvfile:
            temp = csv.reader(csvfile)
            mfcc_lock.acquire()
            for i, roew in enumerate(temp):
                x = []
                for j in range(len(roew)):
                    y = roew[j].split('[')[1].split(']')[0].split(',')
                    x.append(y)
                mfcc_queue.put(x)
                if (mfcc_queue.qsize() < 16):
                    pass
                else:
                    mfcc_lock.release()
                    print("mfcc:" + str(i))
                    while True:

                        if (mfcc_queue.qsize() == 0):
                            mfcc_lock.acquire()
                            break
        mfcc_lock.release()



    def readcsvdata_label(self,filename, label_queue, label_lock):
        print(threading.current_thread().getName())
        with open(filename, 'r') as csvfile:
            temp = csv.reader(csvfile)
            label_lock.acquire()
            for i, roew in enumerate(temp):
                label_queue.put(roew)
                if (label_queue.qsize() < 16):
                    pass
                else:
                    label_lock.release()
                    print("label:" + str(i))
                    while True:

                        if (label_queue.qsize() == 0):
                            label_lock.acquire()
                            break

        label_lock.release()



    # def Get_MFCC(self,wav_file_list):
    #     mfcc_list=[]
    #     i=0
    #     for file_path in wav_file_list:
    #         print(i)
    #         wav,fre=librosa.load(file_path,mono=True)
    #         mfcc=np.transpose(librosa.feature.mfcc(wav,fre),[1,0])
    #         mfcc_list.append(mfcc.tolist())
    #         i+=1
    #     return np.array(mfcc_list)
    def Data_process(self):
        if os.path.exists(self.save_labels) and os.path.exists(self.save_mfcc) and os.path.exists(self.save_word_class_dict):
            # print("Loading data from txt")
            self.label_class_dict=self.Loaddata(self.save_word_class_dict)
            count = -1
            with open(self.save_mfcc, 'rU') as readline:
                for count, line in enumerate(readline):
                    pass
                count += 1
                self.wav_max_len=count

            count = -1
            with open(self.save_labels, 'rU') as readline:
                for count, line in enumerate(readline):
                    pass
                count += 1
                self.label_max_len = count
            #
            # print(count)
            # self.train_mfcc=self.Loaddata(self.save_mfcc)
            # self.train_label=self.Loaddata(self.save_labels)
            #
            # #self.train_mfcc = np.array(self.Loaddata(self.save_mfcc)).reshape((len(self.train_mfcc),-1,20))
            # print("Loading sucessful")
        else:
            original_wav,original_label=self.Get_original_train_data()
            self.encode_dict=self.Get_label_class(original_label)
            self.decode_dict={v:k for k,v in self.encode_dict.items()}
            self.label_class_dict=self.encode_dict
            print(len(self.label_class_dict))
            label_to_num=lambda word:self.label_class_dict.get(word,0)
            self.train_label=[list(map(label_to_num,label)) for label in original_label]
            self.train_mfcc=self.Get_MFCC(original_wav)
            self.wav_max_len = self.Get_max_len_from_list(self.train_mfcc)
            self.label_max_len = self.Get_max_len_from_list(self.train_label)
            for item in self.train_mfcc:
                while len(item) < self.wav_max_len:
                     item.append([0] * 20)
            for item in self.train_label:
                while len(item) < self.label_max_len:
                    item.append(-1)
            self.Savedata_csv(self.save_mfcc,self.train_mfcc)
            self.Savedata(self.save_word_class_dict ,self.label_class_dict)
            self.Savedata_csv(self.save_labels,self.train_label)
            del self.train_label[:]
            del self.train_mfcc[:]


        print(str(self.wav_max_len))
        print(str(self.label_max_len))


    def Get_next_batch(self,batch_size,read_new):
        label_ok=False
        mfcc_ok=False
        batch_label=[]
        batch_wav=[]
        if read_new:
            print("New reader")
            self.start_read_data()
        else:
            pass
        while  mfcc_ok==False or label_ok==False:
            if(label_ok==False):
                if (self.label_queue.qsize() == 16):
                    self.label_lock.acquire()
                    for i in range(16):
                        batch_label.append(self.label_queue.get())
                    self.label_lock.release()
                    label_ok=True
            if (mfcc_ok == False):
                if (self.mfcc_queue.qsize() == 16):
                    self.mfcc_lock.acquire()
                    for i in range(16):
                        batch_wav.append(self.mfcc_queue.get())
                    self.mfcc_lock.release()
                    mfcc_ok = True
            time.sleep(0.2)

        return batch_wav,batch_label

    def start_read_data(self):
        self.label_queue.queue.clear()
        self.mfcc_queue.queue.clear()
        self.read_label_threading = threading.Thread(target=self.readcsvdata_label, args=(self.save_labels,self.label_queue, self.label_lock))
        self.read_mfcc_threading = threading.Thread(target=self.readcsvdata_mfcc, args=(self.save_mfcc,self.mfcc_queue, self.mfcc_lock))
        self.read_label_threading.start()
        self.read_mfcc_threading.start()






    def decod(self,input):
        str=[]
        for item in input.eval():
            str.append(self.decode_dict[item])
        return str
        label_to_num=lambda word:self.label_class_dict.get(word,0)
        self.train_label=[list(map(label_to_num,label)) for label in original_label]







