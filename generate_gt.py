
import pandas as pd
import os
import numpy as np

# Cria ground-truth
PATH_TO_GT_FILES = 'ground-truth/' 
if not os.path.exists(PATH_TO_GT_FILES): 
    os.makedirs(PATH_TO_GT_FILES)

data = pd.read_csv("ground_truth.csv",names=['filename','width','height','class','xmin','ymin','xmax','ymax']) 
unicos = pd.unique(data["filename"])

n=len(unicos)
i=0
for image_path in unicos: 
    i+=1
    print(i,n)
    df = data[data.filename.isin([image_path])]
    #     image_path=image_path.split('/')[1] #Se for rodar no augmentado descomentar isso
    image_path=image_path.split('.')[0]+'.txt'
    
    f = open(PATH_TO_GT_FILES+image_path,'w+')
      
    for index, row in df.iterrows():
        text = 'robot {} {} {} {}\n'.format(int(row.xmin),
                                            int(row.ymin),
                                            int(row.xmax),
                                            int(row.ymax))
        f.write(text)
    f.close()