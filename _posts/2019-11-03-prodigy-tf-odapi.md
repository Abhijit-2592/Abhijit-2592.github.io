---
layout: post # needs to be post
title: Active learning workflow with Tensorflow's object Detection API and Prodigy
featured-img: prodigy #optional - if you want you can include hero image
date: 2019-09-10
published: true
comments: true
---

## NOTE:
The post can also be seen at the official [Prodigy Support Forum](https://support.prodi.gy/t/integrating-tensorflows-object-detection-api-with-prodigy/1965).


I am really excited to share my work of integrating [Tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) with Prodigy which, I did during this summer in collab with [@honnibal](https://github.com/honnibal) and [@ines](https://github.com/ines). You can find the source code in [prodigy-recipes](https://github.com/explosion/prodigy-recipes/tree/master/image/tf_odapi) repo. **The support is still experimental and feedbacks are welcome!** Basically, the point of this post is to act as a guide for the recipe.

##  NOTE:
Since we cannot control how the Tensorflow's Object Detection API changes in the future. I have a fork of the repository which ensures that this recipe works even if some breaking changes happen in the future. The fork can be found [here](https://github.com/Abhijit-2592/models)

# A high-level introduction:
Before getting into the semantics of the recipe, let's first understand how the recipe works at a high level.  A simple working example will be given in the next section.

**To run this recipe you will need [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving).**

`prodigy image.trainmodel -F $PATH_TO/image_train.py --help` to see the **arguments** for the recipe

Running this recipe will create the following 3 folders if not already present:
*   An **export** directory where the models used by Tensorflow Serving will be saved. Specified by `export_dir` argument.
*   A **model** directory where trained model checkpoints and Tensorboard events are stored. Specified by `model_dir` argument.
*   A **data** directory where the **TF-Records** for training are stored. Specified by `data_dir` argument.

## Recipe Flow:
The general flow of the recipe is as follows:

1. Create the object detection model as given in the pipeline.config and convert is as a **custom** [Tensorflow Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator)
2. Check if **model** directory has a checkpoint (if resuming annotations) else, do a dummy training for 1 step. The dummy one step training is required because, the [Tensorflow Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) does not allow **SavedModel** creation without having a checkpoint in **model** dir
3. Save the model as SavedModel in the **export** directory.
4. Start [Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving) and point it to **export** directory so that it can load updated models automatically for **predictions**
5. Perform assisted annotations in prodigy with predictions coming from **Tensorflow Serving**.
6. Use the annotations to train the model in the loop and optionally run evaluation, save the trained model as a **model.ckpt** in the **model** directory and **SavedModel** in **export** directory.
7. Run the **garbage collector**.
8. **Tensorflow Serving** automatically picks up the recent model present in the **export** directory and downs the previous model.
9. Repeat 4 and 5 until satisfied.

In a nutshell, **predictions** happen in **Tensorflow Serving** and the training happens parallely inside **Prodigy**. This structure ensures that, **predictions** can run parallely in a different hardware resource (CPU/GPU) and **training** and **evaluation** can run in another hardware resource(GPU/CPU). **GPU** for **training** and **evaluation** is highly recommended!

## Configuring the recipe:
This section explains how the [pipeline.config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) and other arguments work in coherence for this recipe. This assumes that you have some prior knowledge on how to setup the [pipeline.config](https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs) for Tensorflow's Object Detection API.

* While starting this recipe first time for a new project, make sure to provide a `seed` [TF Record](https://www.tensorflow.org/tutorials/load_data/tf_records) containing **one training** example in **train_input_reader** config in the **pipeline.config**. This is required to do a dummy training for 1 step and save the model as SavedModel in the **export** directory. The dummy 1 step training is required because, the [Tensorflow Estimator](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator) does not allow **SavedModel** creation without having a checkpoint in **model** directory. This **TF Record** can be created from a CSV file using the provided [create_tfrecord.py](https://github.com/explosion/prodigy-recipes/blob/master/image/tf_odapi/misc/create_tfrecord.py) script.
```python
train_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/train.record"
  }
}
```
However, if you are resuming annotations, you can skip the above, iff your **model** directory already has checkpoints from the previous runs.
* If you want to run the **evaluation** also in parallel(set by `run_eval` flag argument) you need to provide the **eval_input_reader** config in the **pipeline.config**.
```python
eval_input_reader {
  label_map_path: "PATH_TO_BE_CONFIGURED/label_map.pbtxt"
  tf_record_input_reader {
    input_path: "PATH_TO_BE_CONFIGURED/validation.record"
  }
}
```
N number of samples are sampled from this **validation.record** (set by `eval_steps` argument) and evaluation is run on these examples. Supports all the [evaluation protocols](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/evaluation_protocols.md) supported by the Object Detection API

## Logging
* Set Prodigy logging level to `basic` to view detailed logs from this recipe.
* Optionally set [Tensorflow Logging](https://www.tensorflow.org/api_docs/python/tf/logging) to 10/20 if you want to see detailed Tensorflow logs. This is set by **tf_logging_level** argument

## Notes and Recommendations
* Object detection algorithms are extremely resource hungry! So, make sure that you run this recipe with **Tensorflow GPU**. However, you can choose to run **Tensorflow Serving** in **CPU** without much loss in performance.
* Point **TensorBoard** to **model** directory to view the training progress. The TensorBoard is really well populated. Especially with **evaluation** enabled.
* The recipe also supports all of the `data augmentations` provided by the Object Detection API out of the box. This can be enabled in the **pipeline_config**. This is especially useful if you are setting the **steps_per_epoch** argument to be more than the number of annotated examples.
* A custom **garbage collector** ensures that only recent N files/folders are stored in the **export** and **data** directory. This is specified by **temp_files_num** argument. The number of recent model checkpoints stored in **model** directory is governed by **max_checkpoints_num** argument.
* It is recommended to provide the  `label_map_path`  in the  **pipeline.config**  rather than passing it as an argument to the recipe

# A simple working example.

Let us try teaching an object detector to detect **Raccoons**. This toy dataset can be found in [this github](https://github.com/datitran/raccoon_dataset) repo. This repo already has the images and annotations(which we don't need for training but needed for evaluation) stored in TF-records and also the image files. I will be running the training/evaluation on **GPU**(a conda environment containing tensorflow-gpu version 1.12.0) and the predictions using **Tensorflow-Serving**(a docker container) running on **CPU**. Additionally, I am using a manually compiled image of Tensorflow Serving so as to use **AVX2** instructions. This optimized image can be downloaded by running `docker pull abhijit2592/tensorflow-serving-devel`. If your CPU does not support **AVX2** instructions, you also use the official image `docker pull tensorflow/serving:latest-devel`.


First, let's setup the following directory structure:
```
raccoon_detection
├── export_dir
│   ├── serve_models.conf
│   ├── serve_models.sh
├── labelmap.pbtxt
├── pipeline.config
├── run_tensorflow_serving.sh
└── run_train.sh
```
**Note**: I am manually creating an export_dir because it's easier to setup Tensorflow-Serving with docker this way.

### Contents of serve_models.sh
```
#!/bin/bash
nohup tensorflow_model_server \
--port=8500 \
--model_config_file=/tensorflow_servables/serve_models.conf >/tensorflow_servables/serving.log
```
### Contents of serve_models.conf
```
model_config_list: {
    config: {
    name: "faster_rcnn_raccoon",
    base_path: "/tensorflow_servables",
    model_platform: "tensorflow"
    }
}
```
### Contents of labelmap.pbtxt
```
item {
  id: 1
  name: 'raccoon'
}

```

### Contents of pipeline.config
```
model {
  faster_rcnn {
    num_classes: 1
    image_resizer {
      keep_aspect_ratio_resizer {
        min_dimension: 600
        max_dimension: 1024
      }
    }
    feature_extractor {
      type: 'faster_rcnn_inception_v2'
      first_stage_features_stride: 16
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
      op: CONV
      regularizer {
        l2_regularizer {
          weight: 0.0
        }
      }
      initializer {
        truncated_normal_initializer {
          stddev: 0.01
        }
      }
    }
    first_stage_nms_score_threshold: 0.0
    first_stage_nms_iou_threshold: 0.7
    first_stage_max_proposals: 300
    first_stage_localization_loss_weight: 2.0
    first_stage_objectness_loss_weight: 1.0
    initial_crop_size: 14
    maxpool_kernel_size: 2
    maxpool_stride: 2
    second_stage_box_predictor {
      mask_rcnn_box_predictor {
        use_dropout: false
        dropout_keep_probability: 1.0
        fc_hyperparams {
          op: FC
          regularizer {
            l2_regularizer {
              weight: 0.0
            }
          }
          initializer {
            variance_scaling_initializer {
              factor: 1.0
              uniform: true
              mode: FAN_AVG
            }
          }
        }
      }
    }
    second_stage_post_processing {
      batch_non_max_suppression {
        score_threshold: 0.0
        iou_threshold: 0.6
        max_detections_per_class: 100
        max_total_detections: 300
      }
      score_converter: SOFTMAX
    }
    second_stage_localization_loss_weight: 2.0
    second_stage_classification_loss_weight: 1.0
  }
}

train_config: {
   data_augmentation_options {
    random_horizontal_flip {
	}
    random_vertical_flip {
	}
  }
  batch_size: 1
  optimizer {
    momentum_optimizer: {
      learning_rate: {
        manual_step_learning_rate {
          initial_learning_rate: 0.0002
          schedule {
            step: 900000
            learning_rate: .00002
          }
          schedule {
            step: 1200000
            learning_rate: .000002
          }
        }
      }
      momentum_optimizer_value: 0.9
    }
    use_moving_average: false
  }
  gradient_clipping_by_norm: 10.0
  fine_tune_checkpoint: "PATH TO COCO TRAINED CHECKPOINT/model.ckpt"
  from_detection_checkpoint: true
}

train_input_reader: {
  tf_record_input_reader {
    input_path: "PATH TO/test.record"
  }
  label_map_path: "PATH TO/labelmap.pbtxt"
}

eval_config: {
  num_examples: 10
}

eval_input_reader: {
  tf_record_input_reader {
    input_path: "PATH TO/test.record"
  }
  label_map_path: "PATH TO/labelmap.pbtxt"
  shuffle: false
  num_readers: 1
}
```
**NOTE:** Here we setup 2 types of data augmentations namely: `random_horizontal_flip` and `random_vertical_flip`. You can also setup other augmentations. An exhaustive list of augmentations can be found [here](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto)

### Contents of run_tensorflow_serving.sh
```
#!/bin/bash
docker run -d \
--name raccoon_tfodapi_container \
-v $PATH TO/export_dir:/tensorflow_servables \
-p 8500:8500 \
abhijit2592/tensorflow-serving-devel:latest \
bash /tensorflow_servables/serve_models.sh
```
### Contents of run_train.sh
```
#!/bin/bash
PRODIGY_LOGGING=basic prodigy image.trainmodel \
-F $PATH TO/image_train.py \
odapi_train \
$PATH TO/raccoon_dataset/images \
$PATH TO/pipeline.config \
localhost \
8500 \
faster_rcnn_raccoon \
--threshold 0.9 \
--run-eval \
--eval-steps 10 \
--steps-per-epoch 100 \
--tf-logging-level 40
```
**NOTE**: Here we are setting --steps-per-epoch to 100. This ensures that even if we annotate only 20 images, the API will sample 100 augmented images from the annotated 20 images. Thus we can synthetically increase the training set on the fly.

# Starting the recipe:
Now we have all the required scripts.
- First, run `bash run_tensorflow_serving.sh` from terminal. This should create a log file named `serving.log` inside the **export_dir** and start the **Tensorflow Serving**. You can use this log file to track how subsequent models are loaded and unloaded. Initially the logfile will show **No versions of servable faster_rcnn_raccoon found under base path /tensorflow_servables**. This is expected because we don't have any servables inside the `export_dir`. The servables will be created when we run the `run_train.sh` script.
- Now we are ready to start `Prodigy`. Run `bash run_train.sh` from terminal. Make sure you are using `tensorflow-gpu` while running this script because this part is extremely resource-intensive.
- Now you can open your browser and start annotating. As soon as you press the `save` button you can see that the training starts as specified in the `pipeline.config` file. You can see live log reports in the terminal where you ran the `run_train.sh` command.
- After annotating a few samples (say 35-40 images), you should start seeing good predictions from the model.
- You can also track the training in **Tensorboard** by running `tensorboard --logdir=$path_to/model_dir`

That's all folks! We have integrated **Tensorflow's Object Detection API** with **Prodigy**.

## PS
- We also have a few other recipes which you might find useful. All of these can be found in [prodigy-recipes](https://github.com/explosion/prodigy-recipes/tree/master/image/tf_odapi).
- We also have a **Speed vs Accuracy** tradeoff study for few models from `Tensorflow model zoo` [here](https://github.com/explosion/prodigy-recipes/blob/master/image/tf_odapi/docs/model_performance.md)
- Also, a few [miscellaneous scripts](https://github.com/explosion/prodigy-recipes/tree/master/image/tf_odapi/misc) which you might find useful.
