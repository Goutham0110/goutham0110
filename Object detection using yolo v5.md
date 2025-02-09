# Object detection using yolov5

One of the fascinating concepts that I admire is object detection. A computer being able to be trained to detect a object/objects is not just fancy to hear but also an interesting process to build. Building a custom object detector and playing with object detection algorithms is much fun than you think (and also projects including object detection and computer vision can come in handy for your resume too). In this article i will explain step-by-step guide to build your own custom object detector.

## Dataset creation
    
  The first step to object detection, without consideration of what algorithm you are going to use, is to create your own dataset. Create an album of photos of the object you want to detect. While taking photos be sure that you get multiple angles and of different lightings of the objects for the model to work well. If you fail to do so, the model may be biased to a certain angle or lighting condition.
    
  1. Sign up to roboflow and click on create new project.
        
       <img src="https://github.com/user-attachments/assets/39489474-2f86-49d1-bfd0-61cc4bafc36d" alt="New Project" width="500" >

        
  2. Upload the image and click on the image to start annotating. After you have annotated all the images click on finish uploading to split the data into test and training.
        
     <img src="https://github.com/user-attachments/assets/28375927-f650-4185-8eae-16f921a437b8" alt="New Project" width="500" >

        
  3. Now go to generate page, here you can apply some pre-processing steps like cropping and other stuff and some augumentation before you can generate your dataset. It’s time to click on the generate button.
        
     <img src="https://github.com/user-attachments/assets/ec37fea0-86c4-4457-8ef4-71c56be67bde" alt="Generate" width="500" >


  4. Click on the export button and choose the format required. If you are following along we will be using yolov5 PyTorch. Roboflow gives you various options on how you can download your dataset. 
        1. You can download a zip file to your machine
        2. You can get a python code which will download your dataset
        3. You can get a terminal command 
        4. You can get a URL
        
        You can choose anything, I mean everything is gonna get your dataset, right? Whichever’s comfortable.
        
        <img src="https://github.com/user-attachments/assets/944710ec-621b-49d6-91e9-0807223d776c" alt="Export" width="500" >


    
  > Roboflow creates a file containing these information which will be used while training and testing our object detection model. 
  You can create dataset for almost any algorithm you choose using roboflow this will be lifesaver for future projects.
  > 
    
## yolov5
    
   There is a high chance that you have already heard about YOLO, which stands for "You Only Look Once". It is the fifth release of the YOLO series of models, created by Ultralytics in 2020. YOLO was the first object detection model to combine bounding box prediction with object classification into a single end-to-end differentiable network. It was written and maintained using a framework called Darknet. As the first YOLO model written in PyTorch, YOLOv5 is lightweight and easy to use. Even so, YOLOv5 did not significantly change the architecture of YOLOv4. It also did not improve accuracy to a noticeable level when compared to YOLOv4 when tested against a common benchmark, COCO. But when compared in terms of size and speed, YOLOv5 has shown better results. It is 88% smaller and also 188% faster than its previous model. YOLOv5 will offer you a much quicker development pace when moving into deployment.
    
## neural network architecture
    
  ```python
    # parameters
    nc: 1  # number of classes  # CHANGED HERE
    depth_multiple: 0.33  # model depth multiple
    width_multiple: 0.50  # layer channel multiple
    
    # anchors
    anchors:
      - [10,13, 16,30, 33,23]  # P3/8
      - [30,61, 62,45, 59,119]  # P4/16
      - [116,90, 156,198, 373,326]  # P5/32
    
    # YOLOv5 backbone
    backbone:
      # [from, number, module, args]
      [[-1, 1, Focus, [64, 3]],  # 0-P1/2
       [-1, 1, Conv, [128, 3, 2]],  # 1-P2/4
       [-1, 3, BottleneckCSP, [128]],
       [-1, 1, Conv, [256, 3, 2]],  # 3-P3/8
       [-1, 9, BottleneckCSP, [256]],
       [-1, 1, Conv, [512, 3, 2]],  # 5-P4/16
       [-1, 9, BottleneckCSP, [512]],
       [-1, 1, Conv, [1024, 3, 2]],  # 7-P5/32
       [-1, 1, SPP, [1024, [5, 9, 13]]],
       [-1, 3, BottleneckCSP, [1024, False]],  # 9
      ]
    
    # YOLOv5 head
    head:
      [[-1, 1, Conv, [512, 1, 1]],
       [-1, 1, nn.Upsample, [None, 2, 'nearest']],
       [[-1, 6], 1, Concat, [1]],  # cat backbone P4
       [-1, 3, BottleneckCSP, [512, False]],  # 13
    
       [-1, 1, Conv, [256, 1, 1]],
       [-1, 1, nn.Upsample, [None, 2, 'nearest']],
       [[-1, 4], 1, Concat, [1]],  # cat backbone P3
       [-1, 3, BottleneckCSP, [256, False]],  # 17 (P3/8-small)
    
       [-1, 1, Conv, [256, 3, 2]],
       [[-1, 14], 1, Concat, [1]],  # cat head P4
       [-1, 3, BottleneckCSP, [512, False]],  # 20 (P4/16-medium)
    
       [-1, 1, Conv, [512, 3, 2]],
       [[-1, 10], 1, Concat, [1]],  # cat head P5
       [-1, 3, BottleneckCSP, [1024, False]],  # 23 (P5/32-large)
    
       [[17, 20, 23], 1, Detect, [nc, anchors]],  # Detect(P3, P4, P5)
      ]
  ```
    
## training model
    
   After placing your dataset in the program folder, you can begin to start training your model. To do that you just have to run the train.py program in the yolov5 folder with the necessary parameters like number of epoch and batch size and others too as command line arguments. 
    
   ```bash
     python train.py --img 416 --batch 64 --epochs 500 --data '/content/data.yaml' --cfg '/content/custom_yolov5s.yaml' --weights ''
   ```
    
  If your model starts training, Good job! You are half way there. After the training completes, which may take a long time(of course it depends on your batch size and number of epochs), you can start testing. First it is a good practice to test the model with your datasets test photo. Tensorboard is a tool that can show the progress of our model training.
    
   ```bash
   tensorboard --logdir '/content/yolov5/runs/train/exp2'
   ```
    
   Running the ```detect.py``` with its command line arguments will give you the result we are expecting
    
   ```bash
   python yolov5/detect.py --weights '/yolov5/runs/train/exp2/weights/best.pt' --img 416 --conf 0.4 --source './test/images'
   ```
    
   If the object is detected, Great! What to do if the object is not detected?
    
   Just train the model with more epochs in case that fails to work, try adding more data or different data.
    
## Detecting ball using webcam
    
   Detecting the required object using webcam just requires you to change the command line argument “--source path/to/file” as “-- source 0”.
   
   ```bash
   python detect.py --weights 'weights/last_yolov5s_custom.pt' --img 416 --conf 0.4 --source 0
   ```
    
   This will work just fine. In case if that fails to work, or you have multiple webcams, or you are using your mobile as webcam using iriun cam, try changing the number from 0 to 1 or 2. Each number will be assigned as an index to each of your camera. 
    
   Now you are detecting the object you wanted. The project is complete.
    
   > You can also use iriun webcam to use your mobile phone as webcam to detect objects. Sounds fun doesn’t it?
   > 
## Conclusion
    
   Object detection has a huge number of applications. After completing this, you can try out projects which requires object detection but where it is not the whole project. For example, I did a ball tracker, where the camera will automatically track a football and always adjusts itself to keep the ball as the center of the frame. Got to say it, it’s not a complete software project, but uses like 85% of software and a very little hardware. If you think about it, how much time have the cameramen at the football field have been faked by Messi? I think you got what I am trying to say. I will try to post an article about the ball tracker soon. Thank you people!
