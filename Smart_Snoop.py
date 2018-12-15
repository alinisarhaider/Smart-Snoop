#-------------------------------------------------------------------------------------------------------------
#   Imports to be used
#-------------------------------------------------------------------------------------------------------------
import tkinter 
import librosa,math,numpy as np 
from tkinter import font as tkfont,filedialog
from scipy.signal import butter, lfilter
import pyaudio,threading,wave,struct
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk
from PIL import Image
from keras.models import load_model
#-------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------------
#Variables to be used
#-------------------------------------------------------------------------------------------------------------

background = 'white'
labelcolor = 'black'


#-------------------------------------------------------------------------------------------------------------
#Main Screen of the Application
#-------------------------------------------------------------------------------------------------------------
top = tkinter.Tk()
top.title("Smart Snoop")
w,h = top.winfo_screenwidth(), top.winfo_screenheight()
top.geometry("%dx%d+0+0" % (w,h))
top.configure(background=background)
top.attributes("-fullscreen", True)

#-------------------------------------------------------------------------------------------------------------
#Images that will be used for the buttons background
#-------------------------------------------------------------------------------------------------------------
srec = Image.open("G:/Ali_bhai/files/srec.png")
ssrec = ImageTk.PhotoImage(srec)

strec = Image.open("G:/Ali_bhai/files/strec.png")
sstrec = ImageTk.PhotoImage(strec)

im = Image.open("G:/Ali_bhai/files/micc.png")
rec = ImageTk.PhotoImage(im)

im1 = Image.open("G:/Ali_bhai/files/sample.png" )
samples = ImageTk.PhotoImage(im1)

im2 = Image.open("G:/Ali_bhai/files/attach.png" )
attach =ImageTk.PhotoImage(im2)

im3 = Image.open("G:/Ali_bhai/files/exit.png" )
exsit = ImageTk.PhotoImage(im3)

recorder = Image.open("G:/Ali_bhai/files/recorder2.png")
recarder = ImageTk.PhotoImage(recorder)

bck = Image.open("G:/Ali_bhai/files/back.png")
backa = ImageTk.PhotoImage(bck)

hm = Image.open("G:/Ali_bhai/files/home.png")
home = ImageTk.PhotoImage(hm)

#-------------------------------------------------------------------------------------------------------------
#Model that is loaded to be used in the system
#-------------------------------------------------------------------------------------------------------------
model = load_model('Model/model-016-0.20-0.94-0.97-F3.h5')

#-------------------------------------------------------------------------------------------------------------
#Customizable fonts that will be used in the system
#-------------------------------------------------------------------------------------------------------------
helv36 = tkfont.Font(family='Stencil', size=30, weight=tkfont.BOLD)
helv38 = tkfont.Font(family='Stencil', size=40, weight=tkfont.BOLD)
helv37 = tkfont.Font(family='Stencil', size=90, weight=tkfont.BOLD)


#-------------------------------------------------------------------------------------------------------------
#Recording Screen 
#-------------------------------------------------------------------------------------------------------------
def record():
    root =tkinter.Toplevel()
    root.title("Smart Snoop")
    w,h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.geometry("%dx%d+0+0" % (w,h))
    root.configure(background=background)
    root.attributes("-fullscreen", True)    
    global is_playing
    global my_thread
    is_playing = False
    my_thread = None
#-------------------------------------------------------------------------------------------------------------
#Method to start recording
#-------------------------------------------------------------------------------------------------------------
    def start_recording():
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        CHUNK = 1024
        global WAVE_OUTPUT_FILENAME
        WAVE_OUTPUT_FILENAME = "recorded.wav"
        
        global audio
         
        audio = pyaudio.PyAudio()
         
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        global frames
        frames = []
         
        while is_playing:
            data = stream.read(CHUNK)
            frames.append(data)         

        stream.stop_stream()
        stream.close()
        audio.terminate()
    
    # --- functions ---
#-------------------------------------------------------------------------------------------------------------
#Method to manage the start recording button
#-------------------------------------------------------------------------------------------------------------
    def press_button_play():
        global is_playing
        global my_thread
        global timethread
        button_stop.config(state = 'normal')
        button_start.config(state = 'disabled')
        if not is_playing:  
            is_playing = True
            my_thread = threading.Thread(target=start_recording)
            my_thread.start()
#-------------------------------------------------------------------------------------------------------------
#Method to manage the stop recording button           
#-------------------------------------------------------------------------------------------------------------
    def press_button_stop():
#        toggle()
        global is_playing
        global my_thread
        
        button_start.config(state = 'normal')
        button_stop.config(state = 'disabled')
    
        if is_playing:
            is_playing = False
            waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
            my_thread.join()
            root.destroy()
            readFile('recorded.wav')
            
#-------------------------------------------------------------------------------------------------------------
#Method to redirect user to main menu           
#-------------------------------------------------------------------------------------------------------------
    def backs():
        root.destroy()
     
#-------------------------------------------------------------------------------------------------------------
#Buttons and lables of the recording screen
#-------------------------------------------------------------------------------------------------------------
    L = tkinter.Label(root, text ="Smart Snoop", bg = background, fg = "black", font = helv37)
    L.pack(padx=2, pady = 20)
    
    EE = tkinter.Button(root, image = recarder , command = sample , bg = background, borderwidth = 0)
    EE.place(x = 500, y = 280,width = 900, height = 500)
    

    button_start = tkinter.Button(root,image = ssrec, command=press_button_play,bg = background,borderwidth = 0)
    button_start.place(x = 800, y = 800,width = 100, height = 100)
    
    button_stop = tkinter.Button(root, image = sstrec, command=press_button_stop ,bg = background,borderwidth = 0)
    button_stop.place(x = 1000, y = 800,width = 100, height = 100)
    button_stop.config(state = 'disabled')
    
    button_EXIT = tkinter.Button(root, image = backa, command=backs ,bg = background,borderwidth = 0)
    button_EXIT.place(x = 50, y = 940,width = 100, height = 100)

#-------------------------------------------------------------------------------------------------------------
#Method to load the pre-attached file as a sample 
#-------------------------------------------------------------------------------------------------------------
def sample():
    readFile('preRecorded/All.wav')

#-------------------------------------------------------------------------------------------------------------
#Method to attach a file from computer 
#-------------------------------------------------------------------------------------------------------------
def attachFile():
    filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("WavFile","*.wav"),("WavFile","*.wav")))
   # print (filename)
    if(len(filename)>0):
        
        readFile(filename)
#-------------------------------------------------------------------------------------------------------------
#Method to rread the file obtained from recording, sample and attach file function
#-------------------------------------------------------------------------------------------------------------
def readFile(filename):
    data,fs = librosa.load(filename, mono=True, sr=44100)
    preProcessing(data,fs)
    
#-------------------------------------------------------------------------------------------------------------
#Pre pROCESSING METHOD
#-------------------------------------------------------------------------------------------------------------
def preProcessing(data,fs):
    global shoot
    global root
    shoot = tkinter.Toplevel()
    shoot.title("Smart Snoop")
    w,h = shoot.winfo_screenwidth(), shoot.winfo_screenheight()
    shoot.geometry("%dx%d+0+0" % (w,h))
    shoot.configure(background=background)
    shoot.attributes("-fullscreen", True)
    
#-------------------------------------------------------------------------------------------------------------
#  Butter Band Pass filter for noise reduction
#-------------------------------------------------------------------------------------------------------------
    def butterBandpass(lowcut, highcut, fs, order=3):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
#-------------------------------------------------------------------------------------------------------------
# Butter bandpass filter 
#-------------------------------------------------------------------------------------------------------------
    def butterBandpassFilter(data, lowcut, highcut, fs, order=3):
        b, a = butterBandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y
    

        
    lowcut = 300.0
    highcut = 3000.0
    
    y = butterBandpassFilter(data, lowcut, highcut, fs, order=3)
    data=y

#-------------------------------------------------------------------------------------------------------------
#Lables, text and buttons of output screen
#-------------------------------------------------------------------------------------------------------------

    FF = tkinter.Label(shoot, text ="Smart Snoop", bg = background, fg = labelcolor, font = helv37)
    FF.pack(padx=2, pady = 30)
    label41 = tkinter.Label(shoot,text="All Words", borderwidth = 0,font=helv38 ,bg =background, fg = 'black')
    label41.place(x = 350, y = 230, width = 300, height = 50)
    
    label42 = tkinter.Label(shoot,text="Suspecious Words", borderwidth = 0,font=helv38 ,bg =background, fg = 'black')
    label42.place(x = 1130, y = 230, width = 590, height = 50)
#-------------------------------------------------------------------------------------------------------------


    
#-------------------------------------------------------------------------------------------------------------
#Method to read the data obtained from pre-processing is sent to segmentation
    segmentation(data,fs,shoot)
#-------------------------------------------------------------------------------------------------------------

    
#-------------------------------------------------------------------------------------------------------------
#Method to segment the speech word by word
#-------------------------------------------------------------------------------------------------------------
def segmentation(data,fs,shoot):
    fd = 0.18
    fl = int(fd*fs)
    N = len(data)
    num_frames = math.floor(N/fl)
    list1 = list()
    list2 = list()
    new_signal = list()
    new=False;
    val=1

    
    for k in range (num_frames):
        frame = list()
        frame.extend(data[k*fl + 1 : fl*(k+1)])
        max_value = max(frame)
        
        if(max_value > 0.02):
            new_signal.extend(frame)
            new=True
        else:
            if(new==True):
                new_signal = np.array(new_signal)
#                librosa.output.write_wav(str(val) + "S.wav", new_signal, fs, norm=False)
                featureExtraction(new_signal,fs,list1,list2)
                val = val+1
                new=False;
                new_signal = list()  


    
 
#-------------------------------------------------------------------------------------------------------------
#Buttons of the output screen + Labels of output screen
#-------------------------------------------------------------------------------------------------------------
    T = tkinter.Text(shoot,bg = '#f5f5f5',fg = labelcolor,font=helv36)
    T.place(x = 105, y = 300,width = 811, height = 500)
    T.insert('end', '   '.join(list1))
    
    T1 = tkinter.Text(shoot,bg = '#f5f5f5',fg = labelcolor,font=helv36)
    T1.place(x = 1000, y = 300,width = 810, height = 500)
    T1.insert('end', ' '.join(list2))
    
#    FFF = tkinter.Label(shoot, text ="Total Number of "+"\n"+"detected words = " + str(len(list1)), bg = background, fg = "black", font = helv36)
#    FFF.place(x = 110, y =834,width = 800, height = 90)
#    
#    FFF1 = tkinter.Label(shoot, text ="Total Number of "+"\n"+"detected suspicious words = " + str(len(list1)), bg = background, fg = "black", font = helv36)
#    FFF1.place(x = 1000, y =834,width = 800, height = 90)
    
    button_EXIT = tkinter.Button(shoot, image=home , command=homess ,bg = background,borderwidth = 0)
                                 
    button_EXIT.place(x = 50, y =940,width = 100, height = 100)

#-------------------------------------------------------------------------------------------------------------
#Method to extract the features from the provided data 
#-------------------------------------------------------------------------------------------------------------
def featureExtraction(new_signal,fs,list1,list2):
    mfccs=librosa.feature.mfcc(new_signal,fs)
    featureReformation(mfccs,list1,list2)


#-------------------------------------------------------------------------------------------------------------
#Method to select the features obtained from the features extraction
#-------------------------------------------------------------------------------------------------------------

def featureReformation(mfcc,list1,list2):
    max_len = 100
    if (max_len > mfcc.shape[1]):
        pad_width = max_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')

    # Else cutoff the remaining parts
    else:
        mfcc = mfcc[:, :max_len]

    classification(mfcc,list1,list2)
    
#-------------------------------------------------------------------------------------------------------------
#Method to classify the words based on selected features 
#-------------------------------------------------------------------------------------------------------------

def classification(mfccs,list1,list2):
    
    allLabels = ['Barood','Blast', 'Bum', 'Fire','Khoon', 
             'Maar', 'Moat', 'Murder', 'Smuggle', 'Taawaan', 'negative']
    
    feature_dim_1 = 20
    feature_dim_2 = 100
    channel = 1
    
    
    
    def get_labels(allLabels):
        labels = allLabels
        label_indices = np.arange(0, len(labels))
        return labels, label_indices
    
    
    
    
    def predict(mfccs, model):
        sample = mfccs
        sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
        
        return get_labels(allLabels)[0][
                np.argmax(model.predict(sample_reshaped))
        ]
    

    output = predict(mfccs, model=model)
    
#    if output != 'negative':
    list1.append(str(output)+ "   ")
    
    if output != 'negative':
        list2.append(str(output)+ "   ")
#-------------------------------------------------------------------------------------------------------------

    
    
#-------------------------------------------------------------------------------------------------------------
#Method to redirect user to main menu
#-------------------------------------------------------------------------------------------------------------
def homess():
    shoot.destroy()
    
#-------------------------------------------------------------------------------------------------------------
#Method to get user out of the system  
#-------------------------------------------------------------------------------------------------------------
def exits():
    top.destroy()
#-------------------------------------------------------------------------------------------------------------



#-------------------------------------------------------------------------------------------------------------
#Script of the live graph on start screen
#-------------------------------------------------------------------------------------------------------------


xar = []
yar = []

style.use('grayscale')
fig = plt.figure(figsize=(19, 5.5), dpi=180)
ax1 = fig.add_subplot(111, frameon = False)
fig.patch.set_facecolor(background)



CHUNK = 1024*2
FORMAT = pyaudio.paInt16
CHANNELS = 1 
RATE = 44100 
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT, 
                channels = CHANNELS, 
                rate = RATE, 
                input = True, 
                output = True, 
                frames_per_buffer = CHUNK
)
abc=256
ax1.set_ylim(0, abc)
ax1.set_xlim(0, CHUNK*2)
ax1.grid(False)
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.axis('off')
ax1.patch.set_alpha(0.5)
x = np.arange(0,2*CHUNK,2)
line, = ax1.plot(x,np.random.rand(CHUNK))

def animate(i):
    data = stream.read(CHUNK)
    data_int = np.array(struct.unpack(str(2*CHUNK)+'B',data),dtype = 'b')[::2] +(abc/2)
    line.set_ydata(data_int)
    yar.append(data_int)
    xar.append(x)


    
#def run():
plotcanvas = FigureCanvasTkAgg(fig, top)
plotcanvas.get_tk_widget().place(x =-300, y = 190,width = 2450, height = 600)
ali=animation.FuncAnimation(fig, animate, interval=2, blit=False)


#-------------------------------------------------------------------------------------------------------------
#buttons and labels of the main screen  
#-------------------------------------------------------------------------------------------------------------
C = tkinter.Label(top, text ="Smart Snoop", bg = background, fg = "black", font = helv37)
D = tkinter.Button(top, image = rec , command = record, bg = background,borderwidth = 0)
E = tkinter.Button(top, image = samples , command = sample , bg = background, borderwidth = 0)
F = tkinter.Button(top, image = attach , command = attachFile , bg = background, borderwidth = 0)                
G = tkinter.Button(top, image = exsit, command = exits , bg = background,borderwidth = 0)                   
DD = tkinter.Label(top, text ="Record",  bg = background,font = helv36 ,fg = labelcolor)          

EE= tkinter.Label(top, text ="   Sample    ", bg = background,font = helv36, fg = labelcolor)

FF = tkinter.Label(top, text ="Attach File ",  bg = background,font = helv36, fg = labelcolor)
                   
GG = tkinter.Label(top, text ="Exit", bg = background,font =  helv36, fg = labelcolor)

                   
C.pack(padx=2, pady = 20)
D.place(x = 140, y = 830,width = 100, height = 100)
E.place(x = 650, y = 830,width = 100, height = 100)
F.place(x = 1180, y = 830,width = 100, height = 100)
G.place(x = 1640, y =830,width = 100, height = 100)

DD.place(x = 98, y = 920,width = 190, height = 50)
EE.place(x = 610, y = 920,width = 190, height = 50)
FF.place(x = 1122, y = 920,width = 260, height = 50)
GG.place(x = 1595, y = 920,width = 190, height = 50)
                
top.mainloop()
#-------------------------------------------------------------------------------------------------------------






