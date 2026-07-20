import numpy as np
import pandas as pd
import os
import glob
import re
import warnings


def loadFile (path):
    import numpy as np
    return np.genfromtxt(path,comments='#')

def saveDat(data,filein):
    import numpy as np
    split=filein.split('.')
    print(data)
    np.savetxt(split[0]+'_TT.'+split[1],data,fmt=['%3.5f','%3.5f','%3.5f'])

# fmt bounds the format of the output file. Old notation %3.5f means three spaces from the left
# so the numbers e.g. 3, 30, 300 has the same alignment. 5 is the number of the decimal places.

def loadFiles(folder,wild): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.dat':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            data.append(np.genfromtxt(str(n),comments='#'))
    return data,filelist,filename

def loadFilesTXT(folder,wild, skiph): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    labels={}
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.txt':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            label=(pd.read_csv(str(n),sep='\t', on_bad_lines='skip', nrows=skiph-1))
            nname=label
            print(nname)
            labels.update({os.path.basename(str(n)) : nname})
            data.append(np.genfromtxt(str(n),comments='#', delimiter='\t', skip_header=skiph))
    return data,filelist,filename,labels

def loadFilesGEN(folder,wild, skiph, form): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    labels={}
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == form:
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            label=(pd.read_csv(str(n),sep='\t', on_bad_lines='skip', nrows=skiph))
            nname=label
            print(nname)
            labels.update({os.path.basename(str(n)) : nname})
            data.append(np.genfromtxt(str(n),comments='#', delimiter='\t', skip_header=skiph))
    return data,filelist,filename,labels

def loadFilesDATXPS(folder,wild): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    labels={}
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.dat':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            label=(pd.read_csv(str(n),sep='\t', engine='python', encoding="ISO8859"))
            nname=label.columns
            print(nname)
            labels.update({os.path.basename(str(n)) : nname})
            data.append(np.genfromtxt(str(n),comments='#', delimiter='\t', skip_header=1))
    return data,filelist,filename,labels

def loadFilesCSV(folder,wild): #folder that should be searched. wild is the constraint for the wild card (* all entries, ? only one)
    data=[] # loaded data
    filelist=[] # list of the datas in data, names with the complete paths
    filename=[] # list of the names only in the folder
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i,ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder,ffile)
    qfilelist=sorted(qfilelist)
    for n in qfilelist:
        if n[-4:] == '.csv':
            filelist.append(str(n))
            filename.append(os.path.basename(str(n)))
            scann=(pd.read_csv(str(n),usecols=[0]))
            values = list(x for x in scann["name"])
            scans=[]
            for i,val in enumerate(values):
                scan=re.findall(r'\d+', values[i])
                scans.append(int(scan[0]))
            data.append(scans)
            data.append(np.genfromtxt(str(n),comments='#', delimiter=',', skip_header=1))
    return data,filelist,filename

def genfromtxt_mpt_robust(path, skip_header, encoding="cp855", delimiter="\t",
                         sniff_lines=200):
    # 1) zjisti rozsah počtu sloupců z prvních N řádků
    counts = []
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for _ in range(skip_header):
            next(f, "")
        for _ in range(sniff_lines):
            line = next(f, "")
            if not line:
                break
            if line.strip() == "":
                continue
            counts.append(len(line.rstrip("\n").split(delimiter)))

    if not counts:
        return np.empty((0, 0))

    min_cols = min(counts)
    max_cols = max(counts)

    # 2) pokud se liší max o 1, ořízni na min_cols (řeší 64/65 případ)
    if max_cols - min_cols <= 1:
        return np.genfromtxt(
            path,
            delimiter=delimiter,
            encoding=encoding,
            skip_header=skip_header,
            usecols=range(min_cols),
        )

    # 3) jinak ruční parsování: pad na NaN na max_cols
    rows = []
    with open(path, "r", encoding=encoding, errors="replace") as f:
        for _ in range(skip_header):
            next(f, "")
        for line in f:
            if line.strip() == "":
                continue
            parts = line.rstrip("\n").split(delimiter)
            # pad / truncate
            if len(parts) < max_cols:
                parts = parts + ["nan"] * (max_cols - len(parts))
            elif len(parts) > max_cols:
                parts = parts[:max_cols]
            # float conversion (MPT bývá numerický)
            try:
                rows.append([float(x) if x != "" else np.nan for x in parts])
            except ValueError:
                # když se objeví nějaký textový řádek, přeskoč
                continue

    return np.array(rows, dtype=float)


def loadFilesMPT(folder, wild):
    data = []
    filelist = []
    filename = []
    
    # Verify if the folder exists before changing directory
    if not os.path.exists(folder):
        raise FileNotFoundError(f"The folder {folder} does not exist.")
    
    os.chdir(folder)
    qfilelist = glob.glob(wild)
    for i, ffile in enumerate(qfilelist):
        qfilelist[i] = os.path.join(folder, ffile)
    qfilelist = sorted(qfilelist)
    
    for n in qfilelist:
        if n.endswith('.mpt'):
            # Check if the file is empty
            if os.path.getsize(n) == 0:
                continue
            
            # Open in text mode (not binary)
            with open(str(n), 'r', encoding="cp855") as f:
                f.readline()
                second_line = f.readline()
                try:
                    nb_header_lines = int(second_line.split(":")[-1].strip())  # More robust extraction
                except (IndexError, ValueError):
                    print(f"Skipping file {n} due to header parsing issue.")
                    continue  # Skip this file instead of raising an error
            
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    file_data = genfromtxt_mpt_robust(str(n), skip_header=nb_header_lines)
                
                # Skip files with irregular data structures
                if file_data.size == 0 or len(file_data.shape) != 2:
                    continue
                
                data.append(file_data)
                filelist.append(str(n))
                filename.append(os.path.basename(str(n)))
            except ValueError:
                print(f"Skipping file {n} due to data inconsistency.")
                continue
    
    return data, filelist, filename
