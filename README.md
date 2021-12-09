# PyTorch-Lightning-Introduction
__*English follows Japanese*__

本リポジトリは、PyTorch Lightningの使用方法を簡易なexampleソースとともに説明するものである。

## 量子化
量子化について、その後の推論の際に、QuantizedCPUで扱えない学習層に注意が必要である。

（例：MnistではQuantizedCPUで扱えない`torch.nn.functional.log_softmax(x, dim=1)`を`forward()`内で使用していたため、除外した。）

PyTorchLightningのLightningModuleにより設計したモデルを三つの手法により量子化した。

PyTorch QAT(QuantizationAwareTraining), PyTorch PTQ(Post Training Quantization), PyTorchLightning QAT callbacks(PyTorch QATの機能を利用したLightningの)の3手法である。

これらの手法で量子化したモデルを用いて、「PyTorchによる推論」と「PyTorch Lightning(Trainer)による推論」を行った。

結果は以下の表の通りである。

| パターン | 量子化手法 | PyTorchによる推論 | PyTorch Lightning(Trainer)による推論 | モデルの保存 |
| :-------------: | ------------- |  ------------- | ------------- |  ------------- |
| ①  | PyToch QAT  | 推論不可 | 推論不可 | 保存可 |
| ②  | PyToch PTQ  | 推論不可 | 推論不可 | 保存可 |
| ③  | PyTorchLightning QAT callbacks  | 推論可 | 推論可 | 保存可 |

# Pytorch-Lightning-Introduction (English)

This repository is for explanation of how to use PyTorch Lightning with simple examples

## Quantization

For quantization, we need to pay attention to the learning layers that cannot be handled by the 'QuantizedCPU' during the subsequent inference.

(e.g., Mnist used unavailable layer, `torch.nn.functional.log_softmax(x, dim=1)`, in `forward()`, so we excluded it.)

The model designed by PyTorchLightning's LightningModule was quantized using three methods.

The three methods are PyTorch QAT (QuantizationAwareTraining), PyTorch PTQ (Post Training Quantization), and PyTorchLightning QAT callbacks(a Lightning's library that uses the functionality of PyTorch QAT).

Using the models quantized by these methods, we performed "inference with PyTorch" and "inference with PyTorch Lightning (Trainer)".

The results are shown in the table below.

| Method | Quantization method | Inference with PyTorch | Inference with PyTorch Lightning(Trainer) | Saving model |
| :-------------: | ------------- |  ------------- | ------------- |  ------------- |
| ①  | PyToch QAT  | Unavailable | Unavailable | Available |
| ②  | PyToch PTQ  | Unavailable | Unavailable | Available |
| ③  | PyTorchLightning QAT callbacks  | Available | Available | Available |

