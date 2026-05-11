# Pneumonia Detection from Chest X-Rays

This project builds a binary image classifier to detect pneumonia from chest X-rays. The aim was to develop a model that reliably flags pneumonia cases while keeping false negatives as low as possible, given the cost of missing a positive diagnosis. Three pretrained models were evaluated: ResNet34, ResNet50, and DenseNet121, with the final model achieving 97% accuracy and 99% pneumonia recall on a held-out test set.

## Data

The dataset was pulled from [mnenendezg/pneumonia_x_ray](https://huggingface.co/datasets/mnenendezg/pneumonia_x_ray) on HuggingFace, which is derived from the Kaggle chest X-ray pneumonia dataset. It contains labeled X-rays split into NORMAL and PNEUMONIA classes with a roughly 3:1 class imbalance.

## Cleaning


The original dataset came pre-split into train, validation, and test sets, but those splits contained overlapping images across partitions. All three were concatenated and deduplicated by hashing raw image bytes using Polars. This produced a clean dataset with no cross-split leakage, which was reshuffled with a fixed seed and resplit 70/20/10 (train/val/test). Images were renamed to their hash values and saved as PNGs.

## Modeling

All models were trained using fastai with ImageNet pretrained weights, fp16 mixed precision, and one-cycle learning rate scheduling. Augmentation followed parameters from published pneumonia detection literature: no flips (X-rays have fixed orientation), up to 20 degrees of rotation, zoom up to 1.2x, warp up to 0.2, and lighting variation up to 0.2. Images were resized to 224x224. `SaveModelCallback` was used throughout to capture the best validation loss checkpoint rather than the final epoch.

Between ResNet34, ResNet50, and DenseNet121, DenseNet121 with frozen pretrained weights outperformed both ResNet variants across all metrics. Notably, unfreezing the DenseNet121 backbone actually degraded performance in multiple tests, suggesting that the pretrained features are well suited for X-ray imagery. 

| Model | Val Loss | Accuracy | Recall | F1 |
|---|---|---|---|---|
| ResNet34 | 0.189 | 93.5% | 92.3% | 95.4% |
| ResNet50 | 0.169 | 94.4% | 94.0% | 96.1% |
| DenseNet121 (frozen only) | **0.163** | **96.3%** | **95.8%** | **97.4%** |

Final inference used test-time augmentation (TTA) at a 0.5 classification threshold. Threshold tuning from 0.2 to 0.5 showed the model produces highly confident predictions, making threshold choice less impactful than expected.

## Results

Final evaluation was run once on the held-out test set using the frozen DenseNet121 checkpoint with TTA.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| NORMAL | 0.97 | 0.93 | 0.95 | 147 |
| PNEUMONIA | 0.97 | 0.99 | 0.98 | 392 |
| **Overall (weighted)** | **0.97** | **0.97** | **0.97** | **539** |

Test performance was consistent with validation performance, indicating the model generalizes well. Misclassification analysis found no systematic visual patterns in the errors, suggesting the remaining failures reflect ambiguity rather than a correctable model or augmentation issue.

## Future Work

Given more time, I'd explore increasing the scope with additional datasets, and shifting from a binary classification model to a more general classification modle that incorporates other conditions.
