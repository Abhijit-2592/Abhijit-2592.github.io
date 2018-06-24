---
layout: post # needs to be post
title: Integrating Keras with Tensorflow Object Detection API # title of your post
featured-img: keras_tfodapi
date: 2018-06-24
published: true
comments: true
---

### Introduction:
Researchers at Google democratized Object Detection by making their [object detection](https://github.com/tensorflow/models/tree/master/research/object_detection) research code public. This made the current state of the art object detection and segementation accessible even to people with very less or no ML background. This post does **NOT** cover `how to basically setup and use the API` There are tons of blog posts and tutorials online which describe the basic usage of the API. ([This](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) post is my favourite and one of the earliest posts on how to use basically setup and use the API). So what the HECK is this post then??? Well ... I have two main objectives in this post:
1. Integrating Keras with Tensorflow Object Detection API:
2. Defining your own model. There is a very sparse [official doc](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/defining_your_own_model.md) that explains it but we will go thourgh it in a bit more detail.
We will accomplish both of the above objective by using **Keras to define our VGG-16 feature extractor for Faster-RCNN**.
3. **Bonus**: Converting an image classification model trained in Keras into an object detection model using the Tensorflow Object Detection API.

**NOTE**:
* This post assumes that you are basically familier with the API.
* You will need Tensorflow version > 1.4.0.
* Some basic understanding of Faster RCNN Architecture. Check this [post](https://lilianweng.github.io/lil-log/2017/12/31/object-recognition-for-dummies-part-3.html) for a brief overview.
* This assumes that your Object detection API is [setup](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) and can be used. Optionally you can use this `python` script to setup the API

``` python
"""python-3
place this code in a file named: 'setup_TF_ODAPI.py' and put this file inside the parent directory
containing the object_detection directory i.e place it in your_path/models/research

Usage:
$python setup_TF_ODAPI.py

follow the instructions on the screen
"""
import os
import subprocess

print("Checking if the environment is already setup ...")
c = subprocess.call(["python" ,"object_detection/builders/model_builder_test.py"],
                    stdout=open(os.devnull, 'wb') # make it quiet
                    )
if c == 0:
    print("environment is already setup")
    print("The Tensorflow Object detection API is ready to use!")

else:
    print('Environment is not setup already. Setting up a fresh environment...')
    curr_work_dir = os.getcwd()
    # compile the protobuf files
    cmd = "protoc object_detection/protos/*.proto --python_out=."
    os.system(cmd)
    print("Protobuf files are compiled")
    # adding environment variable to ~/.bashrc file
    home = os.path.expanduser("~")
    with open(os.path.join(home,".bashrc"),"a") as f:
        f.write("\n")
        f.write("# For Tensorflow object detection API")
        f.write("\n")
        f.write("export PYTHONPATH=$PYTHONPATH:{}:{}/slim".format(curr_work_dir,curr_work_dir))
    print("Python path added to ./bashrc file")

    print("The environment is setup. From a new terminal try running the same script to verify")
```
* Run the script `$python object_detection/builders/model_builder_test.py`. It should run **15** tests (as on 24th June 2018) and print **OK**.

**You can Use this tutorial as a reference to convert any image classification model trained in keras to an object detection or a segmentation model using the Tensorflow Object Detection API the details of which will be given under the bonus section**

# Killing two birds with a single stone!

We will accomplish our two main objectives together! Integrating Keras with the API is easy and straight forward. we can write our keras code entirely using `tf.keras` instead of `tf.contrib.slim` Because, Keras is a part of core Tensorflow starting from version **1.4**.
Now let us build the VGG16 FasterRCNN architecture as given in the [official paper](https://arxiv.org/pdf/1506.01497.pdf). We will follow a three step process to accomplish this.

1. Define the VGG16 FasterRCNN feature extractor inside object/detection/models using tf.keras
2. Registering our model with the API.
3. Writing a small test to check if our model builds and works as intended.

## Step1 : Defining the VGG-16 Faster-RCNN Feature extractor:

Create a file named **faster_rcnn_vgg_16_fixed_feature_extractor.py** inside the `object_detection/models/` and add the following lines to it. The code is straight forward  and self explanatory assuming that you are familier with the Faster-RCNN architecture. We don't need to define the RPN layer because, The RPN layer is dynamically generated by the API and can be configured via the configuration file used while training.

```python

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 10:57:36 2017

@author: abhijit
"""

import tensorflow as tf
import tensorflow.contrib.slim as slim # only for dropout, because you need different behaviour while training and testing
from object_detection.meta_architectures import faster_rcnn_meta_arch

# Define names similar to keras from tf.keras so that ou can copy paste your model code :P
Conv2D = tf.keras.layers.Conv2D
MaxPooling2D = tf.keras.layers.MaxPooling2D
Dense = tf.keras.layers.Dense

class FasterRCNNVGG16FeatureExtractor(faster_rcnn_meta_arch.FasterRCNNFeatureExtractor):
    """VGG-16 Faster RCNN Feature Extractor
    """
    def __init__(self,
               is_training,
               first_stage_features_stride,
               batch_norm_trainable=False,
               reuse_weights=None,
               weight_decay=0.0):

        super(FasterRCNNVGG16FeatureExtractor, self).__init__(
        is_training, first_stage_features_stride, batch_norm_trainable,
        reuse_weights, weight_decay)

    def preprocess(self, resized_inputs):

        """Faster R-CNN VGG-16 preprocessing.

        mean subtraction as described here:
        https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md

        Args:
          resized_inputs: A [batch, height_in, width_in, channels] float32 tensor
            representing a batch of images with values between 0 and 255.0.

        Returns:
          preprocessed_inputs: A [batch, height_out, width_out, channels] float32
            tensor representing a batch of images.

        """
        #  imagenet bgr mean values 103.939, 116.779, 123.68, taken from keras.applications
        channel_means = [123.68, 116.779, 103.939]
        return resized_inputs - [[channel_means]]

    def _extract_proposal_features(self, preprocessed_inputs, scope):
        """Extracts first stage RPN features.

        Args:
          preprocessed_inputs: A [batch, height, width, channels] float32 tensor
            representing a batch of images.
          scope: A scope name. (unused)

        Returns:
          rpn_feature_map: A tensor with shape [batch, height, width, depth]

        NOTE:
            Make sure the naming are similar wrt to keras else creates problem while loading weights
        """
        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(preprocessed_inputs)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)

        return(x)


    def _extract_box_classifier_features(self, proposal_feature_maps, scope):
        """Extracts second stage box classifier features

        Args:
        proposal_feature_maps: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, crop_height, crop_width, depth]
        representing the feature map cropped to each proposal.
        scope: A scope name (unused).

        Returns:
        proposal_classifier_features: A 4-D float tensor with shape
        [batch_size * self.max_num_proposals, height, width, depth]
        representing box classifier features for each proposal.

        Use tf.slim for dropout because you need different behaviour while training and testing
        """
        x = Dense(4096, activation='relu', name='fc1')(proposal_feature_maps)
        x = slim.dropout(x, 0.5, scope="Dropout_1", is_training=self._is_training)
        x = Dense(4096, activation='relu', name='fc2')(x)
        proposal_classifier_features = slim.dropout(x, 0.5, scope="Dropout_2", is_training=self._is_training)

        return(proposal_classifier_features)
```

## Step 2: Registering our model with the API.

In the file `object_detection/builders/model_builder.py` add the following 2 lines:

1. import our model as follows:

`from object_detection.models import faster_rcnn_vgg_16_fixed_feature_extractor as frcnn_vgg16`
2. Append the model to the FASTER_RCNN_FEATURE_EXTRACTOR_CLASS_MAP (it is a python dictionary): `'faster_rcnn_vgg16': frcnn_vgg16.FasterRCNNVGG16FeatureExtractor`

## Step 3: Writing a test to check if our model builds properly:

In the file `object_detection/builders/model_builder_test.py` and the following 2 codes:
1. Import our model as follows:

`from object_detection.models import faster_rcnn_vgg_16_fixed_feature_extractor as frcnn_vgg16`
2. append the following lines before `if __name__ == '__main__':`

``` python

def test_create_faster_rcnn_vgg16_model_from_config(self):
   model_text_proto = """
     faster_rcnn {
       num_classes: 3
       image_resizer {
         keep_aspect_ratio_resizer {
           min_dimension: 600
           max_dimension: 1024
         }
       }
       feature_extractor {
         type: 'faster_rcnn_vgg16'
       }
       first_stage_anchor_generator {
         grid_anchor_generator {
           scales: [0.25, 0.5, 1.0, 2.0]
           aspect_ratios: [0.5, 1.0, 2.0]
           height_stride: 16
           width_stride: 16
         }
       }
       first_stage_box_predictor_conv_hyperparams {
         regularizer {
           l2_regularizer {
           }
         }
         initializer {
           truncated_normal_initializer {
           }
         }
       }
       initial_crop_size: 14
       maxpool_kernel_size: 2
       maxpool_stride: 2
       second_stage_box_predictor {
         mask_rcnn_box_predictor {
           fc_hyperparams {
             op: FC
             regularizer {
               l2_regularizer {
               }
             }
             initializer {
               truncated_normal_initializer {
               }
             }
           }
         }
       }
       second_stage_post_processing {
         batch_non_max_suppression {
           score_threshold: 0.01
           iou_threshold: 0.6
           max_detections_per_class: 100
           max_total_detections: 300
         }
         score_converter: SOFTMAX
       }
     }"""
   model_proto = model_pb2.DetectionModel()
   text_format.Merge(model_text_proto, model_proto)
   model = model_builder.build(model_proto, is_training=True)
   self.assertIsInstance(model, faster_rcnn_meta_arch.FasterRCNNMetaArch)
   self.assertIsInstance(model._feature_extractor,
                         frcnn_vgg16.FasterRCNNVGG16FeatureExtractor)

```

That's it. we have integrated Keras and defined our own Faster RCNN feature extractor simultaneously. Now if you run `$python object_detection/builders/model_builder_test.py`. It should run **16** tests (15 default tests plus our test) and print **OK**. Now you can just add the VGG-16 Faster RCNN Feature Extractor to the .config file as follows:

```
feature_extractor {
      type: 'faster_rcnn_vgg16'
      first_stage_features_stride: 16 #refer the official Faster RCNN paper
    }
```

# Bonus: Converting your Keras classification model to object detection or segmentation model:

Integrating Keras with the API is useful if you have a trained Keras image classification model and you want to extend it to an object detection or a segmentation model. To do that use the above as a guide to define your feature extractor, registering it and writing a test. But the real power is achieved when you are able to use the **Keras classification checkpoint to initialize the object detection or segmentation model**. Since Keras just runs a Tensorflow Graph in the background. Thus you can easily convert any Keras checkpoint to Tensorflow checkpoint. Use the following function to accompolish that.
``` python

"""
@author: Abhijit
"""

from keras.models import model_from_json # or from tf.keras.models import model_from_json
import tensorflow as tf

def keras_ckpt_to_tf_ckpt(keras_model_json_path,
                          keras_weights_path,
                          tf_ckpt_save_path="./tf_ckpoint.ckpt"):
    """ Function to convert a keras classification checkpoint to tensorflow checkpoint
    To use in tensorflow object detection API
    Keyword arguments
    keras_model_json_path --str: full path to .json file (No default)
    keras_weights_path --str:full path to .h5 or hdf5 file (No default)
    tf_ckpt_save_path --str: full path to save the converted tf checkpoint (default = "./tf_ckpoint.ckpt")
    """

    with tf.Session() as sess:
        json_file = open(keras_model_json_path,'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        #load weights
        model.load_weights(keras_weights_path)
        print('loaded keras model from disk')
        saver = tf.train.Saver()
        saver.save(sess,tf_ckpt_save_path)
        print("Tensorflow checkpoint is saved in {}".format(tf_ckpt_save_path))
```

Now change the following lines in the .config file for training and start training!
```
fine_tune_checkpoint: "path to the generated tensorflow checkpoint"
from_detection_checkpoint: false # since ours is a classification checkpoint.
```
VOILA! you have successfully converted your keras classification model to an object detection model using Tensorflow Object Detection API.

# Conclusion:
We have seen
1. How to integrate Keras with Tensorflow Object Detection API.
2. How to define our own Faster RCNN Feature Extractor.
3. How to convert a Keras classification model to object detection model using the API.
