**FaceBoxes con TensorFlow**

# Implementacion con TensorFlow y Keras del paper publicado "FaceBoxes: A CPU Real-time Face Detector with High Accuracy"
 
### Para el entrenamiento se utilizo el dataset "WINDER_FACE".

Como ya hay una implementacion hecha del paper que utiliza Pytorch, este proyecto se basa en hacer una traduccion hacia TensorFlow, 
buscando sus funciones equivalentes.

El proyecto tiene principalmente dos archivos -> train.py y test.py.

# Citation
El paper utilizado para este proyecto
~~~
@inproceedings{zhang2017faceboxes, 
    title = {Faceboxes: A CPU Real-time Face Detector with High Accuracy}, 
    author = {Zhang, Shifeng and Zhu, Xiangyu and Lei, Zhen and Shi, Hailin and Wang, Xiaobo and Li, Stan Z.}, 
    booktitle = {IJCB}, year = {2017} 
}
~~~

### Dificultades que surgieron 
A la hora de correr el proyecto ya hecho en PyTorch surgieron varias dificultades:

- La implementacion fue hecha hace ya algunos años, por lo que utiliza:
    * Python 3.5
    * PyTorch 1.0.0
    * CUDA 11.0 y cudNN 8.0.3
  

 - Se utiliza DataParallel, es decir si se cuenta con dos GPUs o mas, este puede correr en paralelo. A la hora de probar este programa en mi notebook, la cual 
tiene solo una GPU, el codigo me daba error, por lo cual tuve que comentar ciertas partes del codigo: 
~~~
if num_gpu > 1 and gpu_train:
     net = torch.nn.DataParallel(net, device_ids=list(range(num_gpu)))
~~~
