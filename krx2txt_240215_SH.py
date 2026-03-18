import os
import os.path
import numpy as np
import mbs.krx
import glob

# notebook_directory = os.path.dirname(os.path.abspath("__file__"))
# search_pattern = os.path.join(file_directory, notebook_directory.split('/')[-1]+'-MBS-*.krx')
# file_directory = os.path.dirname(os.path.abspath(__file__))
# dir_name = file_directory.split('/')[-1]
# search_pattern = os.path.join('-MBS-*.krx')
krx_files = glob.glob('*.krx', recursive=True)
unprocessed = []
for krx_file in krx_files:
    stemname = os.path.splitext(krx_file)[0]
    txtname1 = stemname + ".txt"
    txtname2 = stemname + "_0.txt"
    if glob.glob(txtname1) or glob.glob(txtname2):
        pass
    else:
        unprocessed += [krx_file]

if krx_files == []:
    print('No file is found')
else:
    if unprocessed ==[]:
        print('All files have been exported')
    else:
        for filename in unprocessed:
            kf = mbs.krx.KRXFile(filename)
            for i in range(kf.num_pages):
                if range(kf.num_pages) == range(0,1):
                    print('export page', i)
                    filename = filename.split('.')[0]
                    out_filename = f"{filename}.txt"
                    kf.export_page_txt(out_filename)
                else:
                    print('export page', i)
                    filename = filename.split('.')[0]
                    out_filename = f"{filename}_{i}.txt"
                    kf.export_page_txt(out_filename, i)
            print("finished exporting", filename,".krx")