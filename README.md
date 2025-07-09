# Car-plate-detection-and-recognition

Project "Car Plate Recognition and Reconstruction with Deep Learning" for the 2024/2025 Computer Vision course.

This repository aims to reimplement the [PDLPR paper](https://www.mdpi.com/1424-8220/24/9/2791) and compare its performance with a CRNN model using CTC loss.

For the detection part, we used a pre-trained YOLOv5 model and fine-tuned it on the CCPD2019 dataset ([link to the pre-trained model](https://huggingface.co/keremberke/yolov5n-license-plate)).  
For the recognition part, we trained the models on the CCPD2019 dataset and then fine-tuned them on CCPD2020 and UC3M-LP to verify their adaptability to different domains and license plate styles.


# Datasets
Original datasets are available at the following links:
- [CCPD2019](https://github.com/detectRecog/CCPD?tab=readme-ov-file#ccpd-chinese-city-parking-dataset-eccv): a collection of Chinese vehicle license plates under various real-world conditions.
- [CCPD2020](https://github.com/detectRecog/CCPD?tab=readme-ov-file#update-on-16092020-we-add-a-new-energy-vehicle-sub-dataset-ccpd-green-which-has-an-eight-digit-license-plate-number): a collection of Chinese green license plates for electric vehicles.
- [UC3M-LP](https://github.com/ramajoballester/UC3M-LP): a collection of annotated European (Spanish) license plates.

We created our own customized versions of the above datasets by selecting and reorganizing the folder structure for training and evaluation purposes. In particular:

- For **CCPD2019**, we used a subset consisting of 30,000 images for training, and 10,000 images each for validation and testing, selected uniformly from each subfolder (Detection and recognition).
- For **CCPD2020**, we used the full dataset (Only recognition).
- For **UC3M-LP**, we only considered license plates of type "A", which correspond to the standard Spanish plate format (Only recognition).

You can download the modified datasets here: [Download datasets](https://drive.google.com/drive/folders/1OFoHWQIxt4oGIwG8GiSzMRre96kkHe7N?usp=drive_link)

# Model Weights

The trained models used for the evaluation can be downloaded from the following links:

**CRNN:**
- [crnn_ccpd2019.pth](https://drive.google.com/uc?id=xxx)
- [crnn_ccpd2020.pth](https://drive.google.com/uc?id=xxx)
- [crnn_uc3m.pth](https://drive.google.com/uc?id=xxx)

**PDLPR:**
- [pdlpr_ccpd2019.pth](https://drive.google.com/uc?id=xxx)
- [pdlpr_ccpd2020.pth](https://drive.google.com/uc?id=xxx)
- [pdlpr_uc3m.pth](https://drive.google.com/uc?id=xxx)


# Project Structure
```
Car-plate-detection-and-recognition/
├── README.md
├── Detection/
│   ├── Finetune_Yolo.ipynb
│   ├── configs/
│       └── ccpd.yaml 
│
├── Recognition/
    ├── Baseline_CCPD2019.ipynb
    ├── Baseline_CCPD2020.ipynb
    ├── Baseline_UC3M_LP.ipynb
    ├── PDLPR_CCPD2019.ipynb
    ├── PDLPR_CCPD2020.ipynb
    ├── PDLPR_UC3M_LP.ipynb
```

# How to run

# Results
## CCPD2019
| Subset          | Seq. Acc. (PDLPR) | Seq. Acc. (CRNN) | Char. Acc. (PDLPR) | Char. Acc. (CRNN) |
|-----------------|------------------|-------------------|-------------------|--------------------|
| CCPD_base       | 99.90%           | 99.64%            | 99.98%            | 99.93%             |
| CCPD_blur       | 70.47%           | 60.97%            | 92.87%            | 86.87%             |
| CCPD_tilt       | 88.16%           | 78.57%            | 97.82%            | 95.60%             |
| CCPD_fn         | 89.13%           | 82.42%            | 97.90%            | 95.72%             |
| CCPD_db         | 79.68%           | 66.27%            | 95.94%            | 90.29%             |
| CCPD_rotate     | 92.04%           | 89.44%            | 98.53%            | 98.02%             |
| CCPD_challenge  | 81.01%           | 72.13%            | 95.91%            | 91.96%             |
| CCPD_weather    | 99.36%           | 98.08%            | 99.89%            | 99.62%             |

## CCPD2020
| Set          | Seq. Acc. (PDLPR) | Seq. Acc. (CRNN) | Char. Acc. (PDLPR) | Char. Acc. (CRNN) |
|-----------------|------------------|-------------------|-------------------|--------------------|
| CCPD_2020       | %           | 90.83%             | %            | 97.08%             |

## UC3M-LP
| Set          | Seq. Acc. (PDLPR) | Seq. Acc. (CRNN) | Char. Acc. (PDLPR) | Char. Acc. (CRNN) |
|-----------------|------------------|-------------------|-------------------|--------------------|
| UC3M-LP         | %           | 92.50%             | %            | 96.63%             |

# Authors
- [A. Infantino 1922069](https://github.com/alessiainf)
- [A. Di Chiara 1938462](https://github.com/AlessandroDiChiara)
- [F. Fragale 2169937](https://github.com/Bannfrost99)

# Acknowledgments
- We thank the creators of the [CCPD](https://github.com/detectRecog/CCPD) and [UC3M-LP](https://github.com/ramajoballester/UC3M-LP) datasets for providing publicly available annotated data.
- We also thank [keremberke](https://huggingface.co/keremberke) for providing the pre-trained YOLOv5 model for license plate detection.
- This work was developed as part of a university project for license plate detection and recognition.

