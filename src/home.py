import streamlit as st


def main():
    st.markdown(
        '''
        <h1 align="center">
            Disease Detection Using Convolutional Neural Networks
        </h1><br>

        
## **Summary**
### Breaking Down the Problem:-
The Problem Statement required us to create a model for detection of Brain Tumour. Brain Tumour is usually detected by a Doctor by observing MRI Scan of the brain,after which several sophisticated medical tests such as biopsy are to confirm the tumour [1]. So we decided to use the MRI scan as input data for our model to predict whether the patient has a brain tumour or not, futhermore, we decided to classify the tumour if we get the data.
### Finding Dataset:-
We searched on Google for a suitable dataset and found a few datasets containing MRI scans of the brain labeled with various types of tumour. First one we found contained 1200 images in total and after scouring around the internet a little bit we found a dataset containing around 7000 images, belonging to 4 different classes which are No Tumour, Meningioma, Glioma and Pituitary, out of which 5700 were for training and 1300 for testing [2].
### Selecting Appropriate tools and Preprocessing Data:-
After we found the data we uploaded it to Google Drive and used it from there by mounting the drive onto a Colab notebook. We used Google Colaboratory since it allows us to use a GPU which makes training faster and allows us to work collectively on the model. We used Pillow module for data preprocessing and then used the Tensorflow module to make our model. Our Data was organized as images put into folders depending upon the class of tumour they belonged to. We made csv files containing information about the image name, image tumour name, and the label encoding of the tumour for both of the training and testing datasets using the Pandas module. We then used the csv files to load the images into RAM and then preprocessed these images by converting them into RGB and into (224.224) shape.
### Learning About Machine Learning, Deep Learning and Transfer Learning:-
We then learned about basic Machine Learning techniques in scikit-learn [3] and slowly moved to learning about Deep Learning and CNNs in Tensorflow [4]. We also came across Transfer Learning Technique which is very popular in Image Classification problems and learning about it through some articles [5] [6].
### Making the Model:-
So now, equipped with knowledge of Machine Learning, Deep Learning and Transfer Learning, We set out to write our model,We used pretrained model to extract features from the images and then added hidden layers for classification. We tried out several different pretrained models for the feature extraction layer such as ResNet50, InceptionV3 , EfficientNetB2 etc and compared their results to select the best for our model. MobileNetV2 turned out to give the best results and gave best results out of all of them.

### Hosting our Model:-
We decided to make a website for showcasing our model. We looked through different options such as flask, Django and some Javascript frameworks but ended up using Streamlit module because it is very easy to use and beginner friendly, moreover it makes deploying the website super easy.

### Additional Models:-
Since we still had time left, we decided to use the same architecture to tackle similar problems such as Covid Detection and Pneumonia Detection using X-Ray scans of chest etc.
## **1.Introduction:**
The given problem statement has been divided into four subparts:

Find a Suitable Dataset for training the Model
Preprocess the data for training.
Make and train the Model.
Make a Website for the Model.
## **2.Dataset Characteristics:**
To create a model to predict brain tumour, we would need some form of data which could be used to identify the tumour. In the medical field, MRI(Magnetic Resonance Imaging) Report is what the doctors use to identify brain tumour, So we decided to look for an Image Dataset which contained labeled images of MRI Scans of the Brain. We found a Dataset on Kaggle,it had 5700 train images and 1300 test images,almost evenly distributed across the following types:
i) No Tumor
ii) Pituitary
iii) Glioma
iv) Meningioma
The images were all mri images which are grayscale.The size of the images was (512X512).

### 2.1 Data Preprocessing
We uploaded the data into a google drive,then accessed the data in google collab,we created a csv file for easier access to the data and for label encoding,then using the csv file we accessed  the images and resized them into the proper image shape of (224,224,3).We use this shape as it is the best suited image shape we can use for the pretrained cnn models we would use for our model.

## **Model**
### Inception  V3
 
The Inception V3 is a deep learning model for image classification that is built on convolutional neural networks. The Inception V3 is an improved version of the Inception V1 basic model, which was presented as GoogleNet in 2014 . Inception V3's model architecture can be seen in Figure 1.
![image](https://user-images.githubusercontent.com/108052351/184403575-031a720e-4412-4659-ba1f-90187bf5212d.png)
<p align="center">
  Fig 1: Model Architecture of Inception V3.
</p>

The model was trained over 20 epochs using the Adam Optimizer with a learning rate of 0.001 and a loss function of sparse cross categorical entropy. As it can observed in Fig.2, model accuracy increased steadily from 74% to 84% and the accuracy stabilized around 87% in 20 epochs.
![ba103912-8b3b-42d6-8a52-d115468a1487](https://user-images.githubusercontent.com/104026985/184408983-3bf79532-3938-414b-9855-f2f4e155c43f.jpg)
<p align="center">Fig 2: Accuracy for Inception V3 model.</p>

### VGG 19

VGG-19 is a convolutional neural network that is 19 layers deep. You can load a pretrained version of the network trained on more than a million images from the ImageNet database. The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224. VGG 19 model architecture can be seen in Figure 3.

![llustration-of-the-network-architecture-of-VGG-19-model-conv-means-convolution-FC-means](https://user-images.githubusercontent.com/104026985/184412049-86d9deb6-0d0e-4fae-9410-18a66114b6af.jpg)
<p align="center">Fig 3: Model Architecture of VGG 19.</p>
The model was trained over 20 epochs using the Adam Optimizer with a learning rate of 0.001 and a loss function of sparse cross categorical entropy.As it can observed in Fig 4, the model !
accuracy increased steadily from 60% to 80% and the accuracy stabilized around 85% in 20 epochs.

![96cf08d8-f6e0-4007-a49b-1b66e8d82019](https://user-images.githubusercontent.com/104026985/184411396-315438fa-e998-4351-a3c9-6f3a2f4dc5bd.jpg)
<p align="center">Fig 4: Accuracy for VGG 19 model.</p>

### DenseNet 201

DenseNet is a convolutional neural network where each layer is connected to all other layers that are deeper in the network, that is, the first layer is connected to the 2nd, 3rd, 4th and so on, the second layer is connected to the 3rd, 4th, 5th and so on.

![DenseNet-201-Architecture](https://user-images.githubusercontent.com/104026985/184411917-720436fa-8bf7-4a99-8703-549779546d9e.png)
<p align="center">Fig. 5: Model Architecture of DenseNet121.</p>
The model was trained on 20 epochs with adam optimizer at a learning rate of 0.001 and sparse categorical cross entropy as its loss function. As shown in Fig. 6, the model’s accuracy increases steadily from 76% to 84%. The accuracy of this model stabilized around 88% in 20 epochs.

![ee9c4da9-7c94-4efb-9678-b928cd781af4](https://user-images.githubusercontent.com/104026985/184412407-24b02406-cc81-4c2b-b739-ffba8a0a0fa2.jpg)
<p align="center">Fig 6: Accuracy for DenseNet201.</p>

### EfficientNet B2

EfficientNet is a convolutional neural network architecture and scaling method that uniformly scales all dimensions of depth/width/resolution using a compound coefficient. Unlike conventional practice that arbitrary scales these factors, the EfficientNet scaling method uniformly scales network width, depth and resolution with a set of fixed scaling coefficients. For example, if we want to use 2N times more computational resources, then we can simply increase the network depth by αN, width by βN, and image size by γN, where α, β, γ are constant coefficients determined by a small grid search on the original small model. EfficientNet uses a compound coefficient φ to uniformly scales network width, depth, and resolution in a principled way. The base EfficientNet-B0 network is based on the inverted bottleneck residual blocks of MobileNetV2, in addition to squeeze-and-excitation blocks. EfficientNets also transfer well and achieve state-of-the-art accuracy on CIFAR100 (91.7%), Flowers (98.8%), and 3 other transfer learning datasets, with an order of magnitude fewer parameters. Fig.7 shows the model architecture of EfficientNetB2.

![Architecture-of-EfficientNet-B0-with-MBConv-as-Basic-building-blocks](https://user-images.githubusercontent.com/104026985/184412776-ca61c8b5-6404-45f2-b0bc-03848cefffe1.png)
<p align="center">Figure 7: Model Architecture for EfficientNet B2.</p>
The model was trained on 20 epochs with adam optimizer at a learning rate of 0.001 and sparse categorical cross entropy as its loss function. The model’s accuracy fluctuates around 30% as shown in Fig.8 .

![fc0481b2-f640-4f24-b00e-42d52a03caf0](https://user-images.githubusercontent.com/104026985/184412949-693a8d53-0db1-400b-a96b-6296d2d20419.jpg)
<p align="center">Fig 8: Accuracy for EfficientNet B2.</p>

### InceptionResNet V2

InceptionResNetV2 is a convolutional neural network that is trained on more than a million images from the ImageNet database. The network is 164 layers deep and can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. It is an extension of the Inception architecture applied in conjunction with residual blocks. Fig. 9 shows the model architecture of InceptionResNetV2.

![1_CYRgf1i1q_4hx5AcdcaSEg](https://user-images.githubusercontent.com/104026985/184413347-fb0b7429-47bf-4cfe-b0d9-f6078287dfb0.jpg)
<p align="center">Fig 9: Model Architecture for InceptionResNet V2.</p>
The Model is trained on 30 epochs with adam optimizer with learning rate of 0.001 and sparse categorical cross entropy as its loss function. The model’s accuracy barely increases from 74% to 82% and the final accuracy stabilized around 86% as seen in Figure.10 . 

![1d5b573b-a350-41a9-93c9-b341afeaa050](https://user-images.githubusercontent.com/104026985/184413582-de9abbf9-e0f2-4672-afbf-73af9acf8ff1.jpg)
<p align="center">Fig 10: Accuracy for InceptionResNet V2.</p>

### MobileNet V2

MobileNetV2 is a convolutional neural network architecture that seeks to perform well on mobile devices. It is based on an inverted residual structure where the residual connections are between the bottleneck layers.The intermediate expansion layer uses lightweight depthwise convolutions to filter features as a source of non-linearity. As a whole, the architecture of MobileNetV2 contains the initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers.

![The-architecture-of-the-MobileNetv2-network](https://user-images.githubusercontent.com/104026985/184413935-13547c6b-3a42-4952-8911-b75dd192c19f.png)
<p align="center">Fig 11: Model Architecture for MobileNet V2.</p>
The model is trained on 30 epochs with adam optimizer at a learning rate of 0.001 and sparse categorical cross entropy as its loss function.The model’s accuracy increases steadily from 76% to 88% as shown in Fig. 13. Since the model seems to have not stabilized around an accuracy, it was trained for 30 more epochs, the model’s accuracy still steadily increased from 88% to 94%. Since its accuracy was still improving a lot, it was trained for 20 more epochs, then the accuracy seemed to stabilize around 99% as shown in Fig. 14. This architecture is best suited for our model since it achieved 99% accuracy in just 80 epochs whereas other models stabilized around the 85% mark. The model achieved an accuracy of around 99% on the train set and 96% on the test set in just 90 epochs. This is the best architecture for our model and we finalized to use MobileNetV2 as our model.

![image](https://user-images.githubusercontent.com/108052351/184419535-d3e4e7ea-d099-4338-a89a-643259cbd01c.png) 
<p align="center">Fig 12: Classification Report after 80 epochs for MobileNetV2.</p>

![image](https://user-images.githubusercontent.com/108052351/184414912-944d79f9-8aff-4716-8eb4-cce25a65757d.png)
<p align="center">Fig 13: Accuracy for MobileNet V2 during first 20 epochs.</p>

![image](https://user-images.githubusercontent.com/108052351/184414991-1acd8884-6ff9-4dde-bf10-74aeaa573516.png)
<p align="center">Fig 14:Accuracy for MobileNet V2 during last 20 epochs.</p>

## The Website
### Streamlit
We made and deployed the websites using the Streamlit module, it is incredibly easy to use and made our work very easy while making the website [7]. We made a homepage, About Data Page,About us page,Contact Us page and Prediction pages

### Additional Models
Since we were done early, We used the same architecture model and trained it on some other datasets,which are Covid 19 Chest X-Ray, Pneumonia Chest X-Ray , Blood Cells.



## References:
1)	https://www.cancer.net/cancer-types/brain-tumor/diagnosis
2)	https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
3)	https://youtu.be/0B5eIE_1vpU
4)	https://youtu.be/tPYj3fFJGjk
5)	https://towardsdatascience.com/transfer-learning-for-image-classification-using-tensorflow-71c359b56673
6)	https://analyticsindiamag.com/transfer-learning-for-multi-class-image-classification-using-deep-convolutional-neural-network/
7) https://warlord-k-iitisoc-22-ai-ml-21-tumor-detection-appapp-zhwh17.streamlitapp.com/
8) https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset
9) https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
10) https://www.kaggle.com/datasets/paultimothymooney/blood-cells
        ''',
        unsafe_allow_html=True,
    )


if __name__ == '__main__':
    main()
