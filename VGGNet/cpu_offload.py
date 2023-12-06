import torch
import logging


class TagInfo:
    """
    定义 tag 的相关信息

    Args:
        tag (str): Tag 名称
    """

    def __init__(self, tag: str):
        self.tag = tag
        # 该 Tag 是否为第一次 iteration 计算
        self.is_first_iter = True
        # 该 Tag 中中间变量的显存总量
        self.total_activations_memory = 0.0
        # 该 Tag 下已经 offload 的显存量
        self.current_offload_memory = 0.0
        self.stream = torch.cuda.Stream()

    def reset_current_offload_memory(self):
        """
        重置相关变量
        """
        self.current_offload_memory = 0.0
        if self.is_first_iter:
            logging.info(
                f"[{self.tag}] total_activations_memory: {self.total_activations_memory}"
            )
        self.is_first_iter = False


class CPUOffload():
    """
    将标注为 tag 的模型按照 offload_ratio 比例进行中间变量的 H2D 和 D2H，以减少显存的占用。

    Args:
        offload_ratio (float): Offload 的比例
        tag (str): Tag 名称

    Examples:

    .. code-block:: python

        from haiscale.cpu_offload import CPUOffload
        import torch.nn as nn
        import torchvision

        class Net(nn.Module):
            def __init__(self, nhiddens, nlayers):
                super(Net, self).__init__()
                self.model = torchvision.models.resnet50()

            def forward(self, x):
                with CPUOffload(offload_ratio=0.1, tag="resnet"):
                    x = self.model(x)
                return x

    """

    # 记录所有的 Tag 信息
    _tag_infos = {}

    def __init__(self, offload_ratio: float, tag: str):
        super().__init__()
        self.offload_ratio = offload_ratio
        self.tag = tag
        # 低版本的 torch 不做 offload
        if getattr(torch.autograd, 'graph', None) is None:
            self.hook = None
            logging.warning(
                f'Current torch version is {torch.__version__}, not support torch.autograd.graph.saved_tensors_hooks. '
                f'Please upgrade your torch version.')
        else:
            if tag not in self._tag_infos:
                logging.info(f"Regist {tag}, offload ratio is {offload_ratio}")
                self._tag_infos[tag] = TagInfo(tag)
            self.hook = torch.autograd.graph.saved_tensors_hooks(
                self.offload_hook, self.load_hook
            )
            # 用于 H2D 和 D2H 的 stream
            self.stream = self._tag_infos[tag].stream

    def __enter__(self):
        # Tag 注册 Offload 钩子
        if self.hook:
            self.hook.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        # Tag 关闭 Offload 钩子
        if self.hook:
            self._tag_infos[self.tag].reset_current_offload_memory()
            self.hook.__exit__(exc_type, exc_value, traceback)

    @classmethod
    def get_tensor_memory(cls, x: torch.Tensor):
        """
        获取 tensor 的显存占用量

        Args:
            x (str): 需要计算显存的 tensor

        Returns:
            tensor_memory (int): Tensor 的显存占用量

        """
        return x.element_size() * x.nelement()

    def offload_hook(self, x: torch.Tensor):
        """
        按照 offload_ratio 的比例将中间变量从 GPU 上移动到 CPU 上，于 forward 产生中间变量后执行

        Args:
            x (torch.Tensor): 中间变量 tensor

        Returns:
            packed_tuple (tuple): 记录原本的 device 和 offload 后的数据，格式为 (device, offload_x)

        """
        tag_info = self._tag_infos[self.tag]
        tensor_memory = self.get_tensor_memory(x)
        # first forward
        if tag_info.is_first_iter:
            # calculate total activations gpu memory
            tag_info.total_activations_memory += tensor_memory
            packed_tuple = (x.device, x.to("cpu", non_blocking=True))

        # not first forward and should be offload
        elif (
                (tag_info.current_offload_memory + tensor_memory) / tag_info.total_activations_memory
        ) <= self.offload_ratio:
            tag_info.current_offload_memory += tensor_memory
            with torch.cuda.stream(self.stream):
                packed_tuple = (x.device, x.to("cpu", non_blocking=True))

        # not first forward and should not be offload
        else:
            packed_tuple = (None, x)

        return packed_tuple

    def load_hook(self, packed_tuple: tuple):
        """
        将中间变量从 CPU 载入 GPU

        Args:
            packed_tuple (tuple): 记录原本的 device 和 offload 后的数据，格式为 (device, offload_x)

        Returns:
            x (torch.Tensor): 载入原本 device 的中间变量 tensor

        """
        device, x = packed_tuple
        if device is not None and x.device != device:
            event = torch.cuda.Event()
            with torch.cuda.stream(self.stream):
                x = x.to(device, non_blocking=True)
                event.record()
            event.wait()
        return x
 