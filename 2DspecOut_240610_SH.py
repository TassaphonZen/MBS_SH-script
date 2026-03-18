import os
import os.path
import ctypes
import glob
import time
import numpy as np

#================================#

def readspectra(filename):
    
    strN = 0
    file = open(filename,"r")
    
	#Read information
    for i, line in enumerate(file,1):
        #read start of Energy
        if "Start K.E." in line:
            strE=line.split()
            strE=float(strE[2])
        #read step of Energy
        if "Step Size" in line:
            stpE=line.split()
            stpE=float(stpE[2])
        if "NoS" in line:
            NoS=line.split()
            NoS=int(NoS[1])
		#read where to start loading data
        if "DATA:" in line:
            break
    
    #create 2D spectra
    spectra = []

    for i, line in enumerate(file):
        data = line.split()
        data_size = np.size(data)
        
        if data_size == NoS:
            data = list(map(int, data))
        else:
            data = list(map(float, data))[1:]
        
        formatted_line = f"{strE + i * stpE:.7f}    {sum(data):.0f}"
        spectra.append(formatted_line)

    #output data
    fileoutput = open(filename+".2Dspec","w")
    for line in spectra:
        fileoutput.write(line+"\n")
    
    file.close()
    fileoutput.close()
	
#================================#

#filename = input("filename: ")
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
#files=glob.glob(filename+"*.txt")
#print(Qspectra(files[0]))
    
    
#filename = input("waiting for input")
#print('scanning for new files to export')
#os.chdir(os.path.dirname(os.path.abspath(__file__)))
files=glob.glob("*.txt")

starttime = time.time()

number_of_files = 0
exist_files = 0
for data in files:
    exist_files += 1
    
    if not os.path.isfile(data+".2Dspec"):
        print(data+".2Dspec")
        readspectra(data)
        number_of_files += 1

endtime = time.time()

if number_of_files == 0 and exist_files == 0:
    #ctypes.windll.user32.MessageBoxW(0,"File not found!!!","message",0)
    print('No file is found')
elif number_of_files == 0:
    #ctypes.windll.user32.MessageBoxW(0,"All files were already exported!!!","message",0)
    print('All files have been exported')
else:
    #ctypes.windll.user32.MessageBoxW(0,str(number_of_files)+" files has been exported within "+str(format(endtime-starttime,".3f"))+".sec","message",0)
    print(str(number_of_files)+" files has been exported in "+str(format(endtime-starttime,".3f"))+".sec")
    
