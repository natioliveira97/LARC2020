import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import cv2
import os

PATH_TO_SAVED_MODEL = "/home/ubuntu/unbeatables/dataset/LARC2020/exported/my_mobilenet_best/saved_model"


print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))  




def create_inference_file(boxes, confidence, image_path, threshold, h, w):
    index = confidence>threshold
    confidence = confidence[index]
    boxes = boxes[index]
    #     image_path=image_path.split('/')[1] #Se for rodar no augmentado, descomentar isso
    image_path=image_path.split('.')[0]+'.txt'

    f = open(PATH_TO_INFERENCE_FILES+image_path,'w+')
    for i in range(len(confidence)):
        text = 'robot {} {} {} {} {}\n'.format(str(confidence[i]),
                                            int(boxes[i][1]*w),
                                            int(boxes[i][0]*h),
                                            int(boxes[i][3]*w),
                                            int(boxes[i][2]*h))
        f.write(text)
    f.close()
                            

PATH_TO_INFERENCE_FILES = '/home/ubuntu/unbeatables/mAP/input/detection-results/'
if not os.path.exists(PATH_TO_INFERENCE_FILES):
    os.makedirs(PATH_TO_INFERENCE_FILES)


category_index = label_map_util.create_category_index_from_labelmap('/home/ubuntu/unbeatables/dataset/LARC2020/label_dict.txt',use_display_name=True)
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')   # Suppress Matplotlib 

# Inferencia nas imagens normais

# Para rodar no augmentado, alterar e descomentar isso
IMAGE_PATH = '/home/ubuntu/unbeatables/dataset/LARC2020/augumented'
data = pd.read_csv("/home/ubuntu/unbeatables/dataset/LARC2020/augumented/ground_truth.csv",names=['filename','width','height','class','xmin','ymin','xmax','ymax']) 

unicos = pd.unique(data["filename"])


n = len(unicos)
i=0

bb_list = []

for image_path in unicos:
        
    i+=1
    # print('Running inference for {}... '.format(image_path), end='')
    t = time.time()
    image_np = cv2.imread(IMAGE_PATH+'/' + image_path)
    image_np = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

#     print(image_np.shape)
    h = image_np.shape[0]
    w = image_np.shape[1]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)



    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.

    num_detections = int(tf.cast(detections.pop('num_detections'),tf.int32))
    detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
           image_np_with_detections,
           detections['detection_boxes'],
           detections['detection_classes'],
           detections['detection_scores'],
           category_index,
           use_normalized_coordinates=True,
           max_boxes_to_draw=200,
           min_score_thresh=.40,
           agnostic_mode=False)


    bb_list.append(detections['detection_boxes'])
    
    print('tempo  {}/{} = {}'.format(i,n,str(time.time()-t)))

    create_inference_file(detections['detection_boxes'],detections['detection_scores'],image_path,0.35,h,w)
#     #cv2.waitKey(0)



# Cria ground-truth
PATH_TO_GT_FILES = '/home/ubuntu/unbeatables/mAP/input/ground-truth/' 
if not os.path.exists(PATH_TO_GT_FILES): 
    os.makedirs(PATH_TO_GT_FILES)

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






