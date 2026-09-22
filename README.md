# HFS-YOLO

**HFS-YOLO for Multi-Scale Object Detection in UAV Imagery**

Author: **Zhenyu Wang**

HFS-YOLO is a four-scale object detector built on YOLO11s. It combines multi-receptive-field processing, spatial attention, and multi-scale residual processing with a high-resolution P2 prediction branch. The design targets the accuracy–parameter trade-off in aerial imagery containing dense objects and substantial scale variation.

On the **VisDrone2019 validation set**, HFS-YOLO achieves **45.5% mAP50** and **26.9% mAP50–95** with **8.81 million parameters**. Under matched experimental settings on the **DIOR test set**, it achieves **83.2% mAP50** and **62.9% mAP50–95**.

## Method

The detector combines four architectural changes:

| Component | Role |
| --- | --- |
| **FlexiRecepConv** | Combines standard, depthwise, and channel-partial convolutions to process features with multiple receptive fields at the P5 downsampling stage. |
| **ScalableLoc-Link Attention** | Combines triplet and coordinate attention to reweight the refined P3 features. |
| **C3k2NB** | Introduces multi-scale residual processing at selected P4 and P5 fusion stages. |
| **P2 prediction branch** | Adds high-resolution prediction to the existing P3–P5 hierarchy, giving prediction strides of 4, 8, 16, and 32 pixels. |

The manuscript uses `NewConvBlock` and `NewAttention` as the implementation names for FlexiRecepConv and ScalableLoc-Link Attention, respectively.

## Results

The following values are reported in the manuscript. Precision, recall, and mAP are percentages. **Params (M)** denotes millions of model parameters, not model file size in megabytes.

### VisDrone2019 validation set

All variants use matched training and evaluation settings. The ablation sequence is cumulative: each row retains the modules added in the preceding rows.

| Variant | Precision (%) | Recall (%) | mAP50 (%) | mAP50–95 (%) | Params (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 49.3 | 37.6 | 38.5 | 22.9 | 9.42 |
| + FlexiRecepConv | 50.1 | 38.4 | 39.2 | 23.3 | 9.38 |
| + ScalableLoc-Link Attention | 53.4 | 38.5 | 40.2 | 24.0 | 9.42 |
| + C3k2NB | 52.6 | 40.1 | 40.9 | 23.8 | 8.57 |
| **HFS-YOLO (+ P2)** | **54.7** | **43.8** | **45.5** | **26.9** | **8.81** |

Compared with YOLO11s, the full model improves mAP50 by **7.0 percentage points** and mAP50–95 by **4.0 percentage points**, with **6.48% fewer parameters**. These differences use the rounded values shown above.

The `+ C3k2NB` variant is the full model without P2. Adding P2 improves mAP50 by **4.6 percentage points** and mAP50–95 by **3.1 percentage points** in this configuration. This comparison evaluates P2 within the combined architecture; the cumulative sequence does not fully separate interactions among modules.

### DIOR test set

Both models use the same data split, training configuration, checkpoint-selection rule, and evaluation settings. Checkpoints are selected on the validation set before evaluation on the held-out test set.

| Model | Precision (%) | Recall (%) | mAP50 (%) | mAP50–95 (%) | Params (M) |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLO11s | 88.5 | 74.2 | 79.8 | 59.6 | 9.42 |
| **HFS-YOLO** | **89.7** | **77.0** | **83.2** | **62.9** | **8.81** |

HFS-YOLO improves mAP50 by **3.4 percentage points** and mAP50–95 by **3.3 percentage points** over YOLO11s. The DIOR experiment evaluates applicability to an additional dataset; it is not a domain-generalization experiment.

## Environment

The manuscript reports the following environment for the DIOR experiments:

| Item | Version / configuration |
| --- | --- |
| Operating system | Ubuntu 20.04 |
| Python | 3.8.10 |
| PyTorch | 2.0.0 |
| CUDA | 11.8 |
| Ultralytics base version | 8.3.0 |
| GPU | One NVIDIA RTX 3090 |

Use an isolated Python environment and a compatible PyTorch/CUDA installation. Historical installation instructions are available in the [official PyTorch guide](https://pytorch.org/get-started/previous-versions/).

HFS-YOLO requires its custom module implementations and model configuration. Installing the unmodified Ultralytics package alone does not provide these modules.

If the code is distributed as a modified Ultralytics source tree with a `pyproject.toml` or `setup.py`, install it from that repository's root:

```bash
python -m pip install -e .
```

If the code is distributed as separate custom modules, follow their registration and dependency instructions before using the commands below.

## Datasets

Download the datasets through their providers and follow the associated access and usage terms:

- [VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset)
- [DIOR](https://opendatalab.org.cn/OpenDataLab/DIOR)

| Dataset | Training images | Validation images | Test images | Reported evaluation split |
| --- | ---: | ---: | ---: | --- |
| VisDrone2019 detection subset | 6,471 | 548 | Not used for the reported comparison | **Validation** |
| DIOR | 5,862 | 5,863 | 11,738 | **Test** |

For DIOR, optimization uses only the training split. The validation split is used for checkpoint selection, and the selected checkpoint is evaluated once on the test split.

Prepare dataset configuration files with the correct image paths, class names, and split definitions. Preserve the dataset's class ordering and the published split identifiers. For DIOR test evaluation, the dataset configuration must explicitly define the `test` split. The experiments use horizontal bounding boxes.

## Experimental settings

| Setting | VisDrone2019 | DIOR |
| --- | --- | --- |
| Target input size | 640 × 640 | 800 × 800 |
| Training epochs | 200 | 200 |
| Batch size | 16 | 4 |
| Optimizer | SGD | SGD |
| Initial learning rate | 0.01 | 0.01 |
| Momentum | 0.937 | 0.937 |
| Weight decay | 0.0005 | 0.0005 |
| Seed | 0 | 0 |
| Deterministic execution | Enabled | Enabled |
| Mosaic augmentation | Enabled | Enabled; disabled for the final 20 epochs |
| NMS IoU threshold | 0.5 | 0.7 |

For VisDrone2019, the box, classification, and distribution focal loss weights are **7.5 / 0.5 / 1.5**. The checkpoint-selection fitness is:

```text
fitness = 0.1 × mAP50 + 0.9 × mAP50–95
```

Both metrics are expressed as fractions when computing fitness. Precision, recall, and both mAP values in a reported row come from the same selected epoch.

Additional DIOR settings include initialization from pretrained YOLO11s weights, cosine learning-rate decay, a final learning-rate fraction of 0.01, and a nominal batch size of 64. Warm-up lasts five epochs, with momentum 0.8 and bias learning rate 0.1. Early stopping is disabled, and checkpoints are retained at 25-epoch intervals in addition to the best and final checkpoints.

DIOR augmentation uses Mosaic = 1.0, translation = 0.1, scale = 0.5, horizontal flip = 0.5, and vertical flip = 0.5. Rotation, MixUp, Copy-Paste, and multi-scale training are disabled. Automatic mixed precision is enabled, caching is disabled, and eight data-loader workers are used. Validation and test evaluation use confidence = 0.001, NMS IoU = 0.7, and at most 1,000 detections per image.

## Usage templates

The examples below use the Ultralytics command-line interface. **They are templates, not verified repository entry points.** Replace every `/path/to/...` value with an existing local file or directory, and ensure that the custom HFS-YOLO modules are registered before loading the model. See the [Ultralytics 8.3.0 configuration reference](https://github.com/ultralytics/ultralytics/blob/v8.3.0/ultralytics/cfg/default.yaml) for argument definitions.

### Training

Use the complete experiment configuration for the desired dataset:

```bash
yolo detect train \
  model=/path/to/model.yaml \
  data=/path/to/dataset.yaml \
  cfg=/path/to/experiment.yaml
```

The experiment configuration should contain the settings listed above together with the full augmentation, initialization, checkpoint-selection, and validation settings used for that run. Unspecified framework defaults are not a substitute for the original experiment configuration.

For cumulative ablations, change the model configuration while keeping the dataset-specific training and evaluation settings fixed.

### Evaluation on VisDrone2019

Use the validation split and the saved evaluation configuration associated with the experiment:

```bash
yolo detect val \
  model=/path/to/visdrone-checkpoint.pt \
  data=/path/to/visdrone.yaml \
  cfg=/path/to/visdrone-evaluation.yaml \
  split=val imgsz=640 iou=0.5
```

The evaluation configuration must preserve the confidence threshold and other evaluation arguments used for the reported results.

### Evaluation on DIOR

Use the checkpoint selected on the validation set and evaluate the test split:

```bash
yolo detect val \
  model=/path/to/dior-checkpoint.pt \
  data=/path/to/dior.yaml \
  split=test imgsz=800 conf=0.001 iou=0.7 max_det=1000
```

Do not report validation-set values as DIOR test-set results or select checkpoints using test-set performance.

### Inference

```bash
yolo detect predict \
  model=/path/to/checkpoint.pt \
  source=/path/to/images \
  imgsz=640 conf=0.25 save=True
```

Here, `imgsz=640` and `conf=0.25` are example visualization settings. Choose the input size appropriate to the trained model. Visualization thresholds are separate from the thresholds used for quantitative evaluation.

## Interpreting the metrics

- **mAP50** is mean average precision at an IoU matching threshold of 0.50.
- **mAP50–95** averages mAP over ten IoU matching thresholds from 0.50 to 0.95 in increments of 0.05.
- **Params (M)** is the number of model parameters in millions.
- Reported improvements are **percentage-point differences**, unless explicitly identified as relative percentages.

The IoU matching thresholds used to compute AP are distinct from the NMS IoU threshold. Parameter count does not directly measure inference latency, throughput, or activation memory.

Each reported configuration was trained once. The manuscript does not report uncertainty across repeated random seeds, separate size-stratified AP, or measured deployment speedups.

## Research context

The method builds on YOLO11s and established convolution and attention mechanisms, including channel-partial convolution, triplet attention, coordinate attention, and efficient multi-scale attention. The contribution is their integration into the detector and its experimental evaluation. References to these methods and to the datasets are provided in the manuscript.

When referring to this work, use the manuscript title **HFS-YOLO for Multi-Scale Object Detection in UAV Imagery** and author **Zhenyu Wang**. Add the final publication details or persistent identifier when available.

## License

Ultralytics 8.3.0 includes its own [upstream license](https://github.com/ultralytics/ultralytics/blob/v8.3.0/LICENSE). Consult the license files distributed with the released implementation for the terms applicable to its code and dependencies. This README does not assign a new license to the implementation.
