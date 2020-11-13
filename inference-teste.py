import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import tensorflow as tf
import cv2
import os
from PIL import Image
 

## @brief Função auxiliar que calcula a área de um vetor de bounging boxes e retorna um vetor com a área de cada uma.
#  @param left_top Vetor de canto superio esquerdo das boxes.
#  @param right_botton Vetor de canto inferior direito das boxes.
#  @return Retorna um vetor com a área de cada box.
def compute_area_of_array(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

## @brief Função auxiliar que exclui boxes muito parecidas e seleciona dentre elas a que possui a maior confiança
#  @param boxes Vetor de boxes, cada box é um vetor com 4 elementos (x1,y1,x2,y2).
#  @param confidences Vetor de confiança para cada box.
#  @return Retorna um vetor com a filtragem das boxes.
def exclude_similar_boxes(boxes, confidences):
    # Boxes que serão escolhidas.
    chosen_boxes = []
    chosen_confidences = []

    # Percorre o vetor de boxes enquanto ele não estiver vazio.
    while len(boxes)>0:
        # Guarda o index da boz com maior confiaça
        max_confidence_index = np.argmax(confidences)
        # Guarda a box com maior confiaça.
        box = boxes[max_confidence_index]
        confidence = confidences[max_confidence_index]
        # Coloca essa box no vetor de escolhidas.
        chosen_boxes.append(box)
        chosen_confidences.append(confidence)
        # Adiciona uma dimenção para poder usar as próximas funções.
        box = np.expand_dims(box, axis=0)
        # Encontra as coordenadas da interseção entre a box escolhida e todas as outras boxes.
        overlap_left_top = np.maximum(boxes[..., :2], box[..., :2])
        overlap_right_bottom = np.minimum(boxes[..., 2:], box[..., 2:])
        overlap_area = compute_area_of_array(overlap_left_top, overlap_right_bottom)
        area1 = compute_area_of_array(boxes[..., :2], boxes[..., 2:])
        area2 = compute_area_of_array(box[..., :2], box[..., 2:])
        # Calcula o índice jaccard entre a box escolhida e todas as outras boxes da lista.
        jaccard = overlap_area/(area1 + area2 - overlap_area + 0.0000001)
        # Seleciona apenas as boxes que possuem índice jaccard menor que 0.5, logo possuem baixa interseção.
        mask = jaccard < 0.5
        # Retira da lista de boxes as que tiveram muita semelhança.
        confidences=confidences[mask]
        boxes=boxes[mask]

    return chosen_boxes, chosen_confidences



def create_inference_file(boxes, confidence, image_path, threshold, h, w):
    index = confidence>threshold
    confidence = confidence[index]
    boxes = boxes[index]

    # print(f"Boxes antes do filtro: {len(boxes)}, {len(confidence)}")
    boxes, confidence = exclude_similar_boxes(boxes, confidence)
    # print(f"Boxes dps do filtro: {len(boxes)}, {len(confidence)}")

    #     image_path=image_path.split('/')[1] #Se for rodar no augmentado, descomentar isso
    image_path=image_path.split('.')[0]+'.txt'

    f = open(PATH_TO_INFERENCE_FILES+ '/' + image_path,'w+')
    for i in range(len(confidence)):
        text = 'robot {} {} {} {} {}\n'.format(str(confidence[i]),
                                            int(boxes[i][1]*w),
                                            int(boxes[i][0]*h),
                                            int(boxes[i][3]*w),
                                            int(boxes[i][2]*h))
        f.write(text)
    f.close()

def create_inference_files(bb_list):
    for image, bb, h,w in bb_list:
        create_inference_file(bb['detection_boxes'], bb["detection_scores"], image,  0.45, h,w)



def target(unicos, detect_fn, bb_list):
    print("comeco processo")

   
    
    IMAGE_PATH = '/home/ubuntu/test_images'


    i=0
    n = len(unicos)
    for image_path in unicos:
        print(image_path,i)

        t = time.time()
        print(t)
        i+=1
        print(i)
        # print('Running inference for {}... '.format(image_path), end='')
        
        image_np = np.array(Image.open(IMAGE_PATH+'/' + image_path))
        print("carregou img", image_np.shape[0])
        # image_np = cv2.cvtColor(image_np,cv2.COLOR_BGR2RGB)
        input_tensor = tf.convert_to_tensor(image_np)
        print("convert tensor")
        input_tensor = input_tensor[tf.newaxis, ...]
        print("input tensor")
        #     print(image_np.shape)
        h = image_np.shape[0]
        w = image_np.shape[1]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = detect_fn(input_tensor)

        print("detect fn")

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.

        num_detections = int(tf.cast(detections.pop('num_detections'),tf.int32))
        detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
        detections['num_detections'] = num_detections

        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
        print("detections")
        # image_np_with_detections = image_np.copy()

        # viz_utils.visualize_boxes_and_labels_on_image_array(
        #        image_np_with_detections,
        #        detections['detection_boxes'],
        #        detections['detection_classes'],
        #        detections['detection_scores'],
        #        category_index,
        #        use_normalized_coordinates=True,
        #        max_boxes_to_draw=200,
        #        min_score_thresh=.40,
        #        agnostic_mode=False)


        bb_list.append((image_path, detections, h,w))
        # print('tempo  {}/{} = {}'.format(i,n,str(time.time()-t)))

        # create_inference_file(detections['detection_boxes'],detections['detection_scores'],image_path,0.35,h,w)
        #cv2.waitKey(0)
        print("fim ",i)

IMAGE_PATH = '/home/ubuntu/test_images'
PATH_TO_INFERENCE_FILES = '/home/ubuntu/inference'
if not os.path.exists(PATH_TO_INFERENCE_FILES):
    os.makedirs(PATH_TO_INFERENCE_FILES)


category_index = label_map_util.create_category_index_from_labelmap('label_dict.txt',use_display_name=True)
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Process
from threading import Thread

def main():
    

    PATH_TO_SAVED_MODEL = '/home/ubuntu/unbeatables/dataset/LARC2020/exported/my_mobilenet_best/saved_model'

    print('Loading model...', end='')
    start_time = time.time()

    import cv2


    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)
    detect_fn2 = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time)) 


 
    unicos = os.listdir("/home/ubuntu/test_images")
    print(len(unicos))

    print(unicos)

    bb_list = []
    bb_list2 = []
    start_time = time.clock()
    t=time.time()


    p = Process(target=target, args=(unicos[:int(len(unicos)/2)], detect_fn, bb_list,))
    p2 = Process(target=target, args=(unicos[int(len(unicos)/2):], detect_fn2, bb_list2,))
    p.start()

    p2.start()
    p.join()
    p2.join()

    print("Tempo de fim: {}".format(time.time()-t))

    create_inference_files(bb_list)



main()


