# db_faster-r-cnn
# Faster R-CNN

## Some of the code in the project is from：
* https://github.com/pytorch/vision/tree/master/torchvision/models/detection
* https://github.com/WZMIAOMIAO/deep-learning-for-image-processing

## Environment Configuration：
* Python3.6/3.7/3.8
* Pytorch 1.6.0 or above（it is only supported after 1.6.0 using the officially provided hybrid precision training)
* pycocotools(Windows:```pip install pycocotools-windows```)
* Windows or others
* Better with GPU
* For detailed environment configuration :requirements.txt

## File Structure：
```
  ├── backbone: ResNet
  ├── network_files: Faster R-CNN Network including Fast R-CNN and RPN
  ├── train_utils: Training and validation related modules（including cocotools）
  ├── my_dataset.py: Custom dataset for reading VOC datasets
  ├── train_resnet50_fpn.py: resnet50+FPN as a backbone for training
  ├── loss.py:double-balanced loss for classification
  ├── data_division.py:For dividing the test set, training set and validation set
  ├── predict.py: Prediction test using trained weights
  └── pascal_voc_classes.json: pascal_voc Label file
```

## Pre-training weights download address:
* ResNet50+FPN backbone: https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth

## This project uses a dataset in VOC2012 format:

  ├── backbone

​      ├──VOCdevkit

​          ├── VOC2012

​                ├── Annotations(Storing label files)

​                ├── ImageSets(Storing  train.txt and test.txt)

​                ├── JPEGImages(Storing  images jpg or jpeg)

## Train
* Ensure that the data set is prepared in advance

* Ensure that the corresponding pre-trained model weights are downloaded in advance

* Training with train_resnet50_fpn.py

## Faster R-CNN :

![Faster R-CNN](fasterRCNN.png) 

## Double-balanced loss:

![db_loss](https://user-images.githubusercontent.com/76239068/141785572-24c32443-f79a-4702-9be0-4c37ab7f44a4.jpg)

## demo:
![001010](https://user-images.githubusercontent.com/76239068/141785742-2136e3f0-d622-41e1-8222-f3a5953eac59.jpg)
![002129](https://user-images.githubusercontent.com/76239068/141785798-1c0413ef-19d7-49ff-adc6-49052aa2530f.jpg)
![003486](https://user-images.githubusercontent.com/76239068/141785838-f7ef2b06-2886-48a7-8a21-14468c4b21ff.jpg)


