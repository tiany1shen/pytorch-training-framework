# Usage Documentation {ignore}

该文档描述此训练框架的使用方法。本框架在 PyTorch 的基础上对深度学习项目中的各组成部分进行更精确的接口定义。

[TOC]
版本依赖：

```Python
python>=3.11
```

## Dataset

在本框架中，`SizedDataset` 类被用来管理数据集。`SizedDataset` 类为 `torch.utils.data.Dataset` 的映射型子类，内部实现了 `__getitem__` 和 `__len__` 两个方法。在继承 `SizedDataset` 从而实现自定义数据集类的时候，必须重写这两个方法。

- Methods of `SizedDataset` Class:
  
  - `__getitem__(self, index: int) -> Data`：返回经过变换处理的张量数据，类型为 `Data: TypeAlias = torch.Tensor | Sequence[torch.Tensor] | dict[str, torch.Tensor]`。
  
  - `__len__(self) -> int`：返回数据集的长度（映射类数据集应为有限数据集）。

## Neural Network

在本框架中，`NeuralNetwork` 类被用来定义神经网络。`NeuralNetwork` 类继承自 `torch.nn.Module` 类，并在内部实现了 `device` 属性和 `init_weights` 方法。在使用 `NeuralNetwork` 类搭建神经网络时，必须重写 `init_weights` 方法。

- Attributes of `NeuralNetwork` Class:
  
  - `device: torch.device`：返回神经网络实例的权重参数所在的设备。无需重写。

- Methods of `NeuralNetwork` Class:
  
  - `init_weights(self) -> Self`：初始化神经网络实例的权重参数，并返回实例本身。支持链式调用。

## Train & Evaluate

在本框架中，`_Model` 类被用来定义操作神经网络对象的各种方式，比如：计算损失函数值、使用提取的特征进行推断、计算评估模型指标等。在常见的 PyTorch 程序脚本中，这类操作往往是训练代码的一部分，甚至有些时候计算损失函数值会在 `torch.nn.Module.forward()` 内部完成。本框架将这类功能单独抽象出来并且提供以下两个子类分别完成训练时和评估时的相关操作。

### TrainModel Class {ignore}

`TrainModel` 用来管理训练过程中以计算损失函数为核心的各种操作神经网络的方法，其内部实现了 `loss_weights` 属性和 `compute_loss` 方法。对于新的问题和训练过程，要继承此类必须指定 `_loss_weights` 属性并重写 `compute_loss` 方法

- Attributes of `TrainModel` Class:
  
  - `loss_weights: dict[str, float]`：返回一个字典，内部元素键为损失函数的名称，元素值为该损失函数在总损失函数内的贡献权重。无法直接修改，需通过 `_loss_weights` 这个内部属性指定。

- Methods of `TrainModel` Class:
  
  - `compute_loss(self, network: NeuralNetwork, batch: Batch) -> dict[str, ScalarTensor]`：使用指定的批量数据和神经网络计算损失函数值，返回一个字典，其元素键与 `loss_weights` 属性相同，元素值为对应损失函数的带有梯度的标量值。

### EvaluateModel Class {ignore}

`EvaluateModel` 用来管理评估模型过程中操作，与 `TrainModel` 类似，内部实现了 `metrics` 属性和 `evaluate` 方法。要继承此类，必须指定 `_metrics` 属性并重写 `evaluate` 方法。

- Attributes of `EvaluateModel` Class:
  
  - `metrics: list[str]`：返回一个列表，内部元素为评估指标的名称。无法直接修改，需通过 `_metrics` 这个内部属性指定。

- Methods of `EvaluateModel` Class:
  
  - `evaluate(self, network: NeuralNetwork) -> dict[str, float]`：对指定的神经网络进行评估，返回一个字典，其元素键与 `metrics` 属性相同，元素值为对应的指标值。

`EvaluateModel` 对象可以被 `EvaluatePlugin` 插件对象调用，从而将评估过程插入训练循环中。

## Trainer

- [ ] TODO: 补充 Trainer 类的属性和方法

`Trainer` 类为实现训练循环的训练器类，其核心方法为 `__init__`、`loop` 和 `add_plugin` 这三个方法。

- Arguments of `__init__` method:
  - `train_model: TrainModel`：训练器必须调用一个 `TrainModel` 对象来初始化训练过程。
  - `num_epochs`：指定本轮训练的总轮次。
  - `batch_size`：指定训练数据的批次大小，会在后续过程中被更新为 `gradient_accumulation_step` 的整数倍。
  - `gradient_accumulation_step`：梯度累计次数，默认为 1 。
  - `init_seed`：初始随机种子，默认为 `None`，会随机生成一个数作为随机种子值。
  - `device`：指定训练的设备，目前支持 CPU 或 单卡GPU 训练。

- Methods of `Trainer` Class:
  - `loop(self, dataset: SizedDataset, network: NeuralNetwork, optimizer: torch.optim.Optimizer)`：训练循环，使用指定的数据集训练神经网络，参数由优化器进行更新。
  - `add_plugin(self, plugin: Plugin) -> Self`：将 `Plugin` 类型插件注册到训练器内，为训练器添加功能。支持链式调用。

## Plugin

- [ ] TODO: 如何编写自定义插件

本框架中，`Plugin` 插件类被用来向训练器添加额外功能，是支持框架拓展性的核心模块。训练器在训练函数中预留了六个位置对其注册的插件进行回调，通过定义插件在对应位置的功能函数，可以将自定义功能加入训练循环的对应位置。本框架已经实现并可以直接调用的插件类型有：

- LoadCheckpointPlugin
- SaveCheckpointPlugin
- InitializeNetworkPlugin
- LossLoggerPlugin
- MetricLoggerPlugin
- EvaluatePlugin
- ProgressBarPlugin

### Checkpoint-related Plugin {ignore}

通过 `LoadCheckpointPlugin` 和 `SaveCheckpointPlugin` 两类插件，可以实现读取检查点和保存训练检查点的功能。

- `LoadCheckpointPlugin(checkpoint_dir: str)`：读取于 `checkpoint_dir` 这一目录保存的检查点，并在训练开始之前（`Trainer.loop()` 函数内部）对训练器对象的某些内部属性进行覆盖。如果需要使用检查点继续训练，则该插件应该在实例化训练器对象后第一个被注册。
- `SaveCheckpointPlugin(saving_dir: str, saving_period: int)`：每经过 `saving_period` 轮（epoch）的训练，将训练器、神经网络、优化器的状态保存到 `saving_dir / epoch` 这一子目录下。

### Initialize Plugin {ignore}

通过 `InitializeNetworkPlugin` 实现指定训练开始时神经网络参数的功能。

- `InitializeNetworkPlugin(weight_file)`：如果指定了 `weight_file` 文件路径且文件存在，将使用其中保存的模型权重初始化模型，否则将调用 `network.init_weights()` 进行随机初始化。

如果在注册本插件之前已经修改了训练器的 `pretrained_weight_file` 属性（如已经在之前注册 `LoadCheckpointPlugin`）， 将不会覆盖指定的文件目录，及此时 `weight_file` 参数会失效。

### Evaluate Plugin {ignore}

通过 `EvaluatePlugin` 实现在 epoch 结束后对模型进行评估的功能。

- `EvaluatePlugin(eval_model: EvaluateModel, eval_period: int)`：每隔 `eval_period` 轮 epoch，将调用 `eval_model.evaluate(network)` 对神经网络模型进行评估。

### Logger Plugin {ignore}

通过 `LossLoggerPlugin` 和 `MetricLoggerPlugin`，使用 Tensorboard 将训练过程中计算出的损失函数值和评估指标记录下来，可以后续通过 Tensorboard 包可视化并管理实验结果。

- `LossLoggerPlugin(log_dir: str, log_period: int)`：每隔 `log_period` 步（step）训练，记录一次各损失函数的值，日志文件将被保存在 `log_dir` 目录之下。如果损失函数的数量不止一个，还会额外记录一个总损失函数值。
- `MetricLoggerPlugin(log_dir: str, log_period: int)`：每隔 `log_period` 轮（epoch）训练，记录一次评估指标的值，日志文件将被保存在 `log_dir` 目录之下。

### ProgressBarPlugin {ignore}

通过 `ProgressBarPlugin` 在命令行界面打印训练进度条。

- `ProgressBarPlugin(bar_length: int = 10)` 进度条长度为 `bar_length` 个字符，默认长度为 10。
