**FaceBoxes con TensorFlow**

# Implementacion con TensorFlow y Keras del paper publicado "FaceBoxes: A CPU Real-time Face Detector with High Accuracy"

# Citation
El paper utilizado para este proyecto
~~~
@inproceedings{zhang2017faceboxes, 
    title = {Faceboxes: A CPU Real-time Face Detector with High Accuracy}, 
    author = {Zhang, Shifeng and Zhu, Xiangyu and Lei, Zhen and Shi, Hailin and Wang, Xiaobo and Li, Stan Z.}, 
    booktitle = {IJCB}, year = {2017} 
}
~~~


El paper publicado cuenta con dos implementaciones actualmente, la [original](https://github.com/sfzhang15/FaceBoxes) principalmente escrita en C++, y otra, 
la cual se uso de base, escrita en [PyTorch](https://github.com/zisianw/FaceBoxes.PyTorch)

### Training dataset "WINDER_FACE"
Se utilizo el dataset [Winder Face](http://shuoyang1213.me/WIDERFACE/), el cual posee una licencia APACHE LICENSE, VERSION 2.0

Una vez descargada, se deben ubicar las fotos en 

*$FaceBoxes_ROOT/data/WIDER_FACE/images*


### Dificultades que surgieron  a la hora de correr el proyecto ya hecho en PyTorch 

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

### Dificultades que surgieron  a la hora de pasar a TensorFlow


