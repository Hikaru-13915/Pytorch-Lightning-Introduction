# PyTorch-Lightning-Introduction
__*English follows Japanese*__

本リポジトリは、PyTorch Lightningの使用方法を簡易なexampleソースとともに説明するものである。

## Lighitningの基本的な使い方

本リポジトリ内 Basic_usage_of_lightning.py に基本的なPyTorchLightningの使い方記載している。    
大まかな手順としては、PyTorchと基本的には同じである。

- モデル設計
- デバイスや学習環境の設定
- 学習開始

PyTorchLightningによるモデル学習において、モデル設計が最も大きなステップである。
それに比べ、デバイスや学習環境の設定や学習そのものは非常に簡易な記述で実装することができる。
以下にそれぞれのステップを詳細に説明する。

### モデル設計

モデルの設計は、前述のように非常にPyTorchの記述と似ており、PyTorchでは、`torch.nn.Module`をモデルクラスが継承するのに対し、
PyTorchLightningでは、`pytorch_lightning.LightningModule`を継承する。
モデルの内部で定義すべき項目は以下の通り

```
def __init__(self,args):

def forward(self, x):

def train_dataloader(self):

def val_dataloader(self):

def test_dataloader(self):

def training_step(self, batch, batch_idx):

def test_step(self, batch, batch_nb):

def test_end(self, outputs):

def validation_step(self, batch, batch_idx):

def validation_epoch_end(self, outputs):

def configure_optimizers(self):
```
本スクリプトでは、`def setup(self, stage)`を入れているが、基本的には上記の11ステップを記述しておけば、学習と推論がスムーズに行える。  
`def __init__(self)`では、引数のモデルへの反映のために`args`を加え、`def __init__(self,args)`としている。この`__init__()`と`forward()`は、PyTorchでの記述と同じように設計するだけでよい。  
PyTorchLightningでは、後に紹介するcallback機能や、ログ機能等があり、学習済のモデルに新たにモジュールを用いて変更を加えることがあるので、ベースとなるモデルを独立させ（`torch.nn.Module`による設計で良い）、そのモデルをターゲットとなるモデルに継承させることができる。具体的には、本スクリプトの`LitAutoEncoder`で`self.model=Net()`という風にベースとなるモデル`Net()`を継承することで、モデルを後からカスタマイズできる。  

次に設計で大事なのは、データ準備である。これは非常に容易で、`def train_dataloader(self)`, `def val_dataloader(self)`, `def test_dataloader(self)`でそれぞれの用途に合わせたデータローダーを返り値とし設定するだけ。本モデルでは直接、`torch.utils.data.DataLoader`を使用している。  
次に、学習や推論部分の設定を行う。`training_step()`, `test_step()`, `test_end()`, `validation_step()`, `validation_epoch_end()`の設定をしておけば十分であり、`training_step()`, `test_step()`, `validation_step()`では各イテレーションごとのステップ処理を記述している。もしログ機能を用いるのであれば、残したいログをここで設定する。  
`test_end()`, `validation_epoch_end()`では各エポックごとのロス計算等の設定を行う。  

最後に、`configure_optimizers()`ではオプティマイザーの設定を行う。PyTorchでは、ステップごとにオプティマイザーの処理を記述しているが、PyTorchLightningではモデル内で定義できる。
これらの細かな設定により、PyTorchで記述していたような学習や推論のループは不必要になる。

## 量子化
量子化について、その後の推論の際に、QuantizedCPUで扱えない学習層に注意が必要である。

（例：MnistではQuantizedCPUで扱えない`torch.nn.functional.log_softmax(x, dim=1)`を`forward()`内で使用するため、除外した。）

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

