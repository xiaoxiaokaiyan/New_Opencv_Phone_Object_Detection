# 心得：**Opencv+phone 物体检测**

## News
* this code 
* [pytorch-yolo-v3 phone 物体检测](https://github.com/xiaoxiaokaiyan/New_Tensorflow_AE_VAE_FashionMnist_GAN_WGAN_Anime)


## Dependencies:
* Windows10
* python==3.7.4
* opencv-python==4.3.0.36
<br/>


## Visualization Results
* 笔记本调用手机摄像头
<img src="https://github.com/xiaoxiaokaiyan/New_Opencv_Phone_Object_Detection/blob/main/result1.PNG" width = 50% height =50%  div align=center />

* 摄像头物体识别
<img src="https://github.com/xiaoxiaokaiyan/New_Opencv_Phone_Object_Detection/blob/main/result2.PNG" width = 50% height =50%  div align=center />

&nbsp;
<br/>


## Public Zoo:
* coco.names、yolov3.cfg、yolov3.weights。
  * YOLOv3模型文件 link:[链接：https://pan.baidu.com/s/1M8EVfUZ7NCWV5yJMuK2LbQ 提取码：u41w](https://pan.baidu.com/s/1M8EVfUZ7NCWV5yJMuK2LbQ)
  

## Experience：
### （1）调用手机摄像头
```
    手机下载“IP摄像头APP”，运行程序之前将APP打开，提前修改python代码中手机的IP地址。
```
### （2）代码
```
    cv2.namedWindow("camera",1)
    #开启ip摄像头
    video="http://admin:admin@192.168.1.4:8081/"   #此处@后的ipv4 地址需要修改为自己的地址
    capture =cv2.VideoCapture(video)
 
    num = 0;
    while True:
        success,img = capture.read()
        cv2.imshow("camera",img)    

```  



## References:
* [python+OpenCV+YOLOv3打开笔记本摄像头模型检测](https://blog.csdn.net/weixin_43590290/article/details/100736307)
