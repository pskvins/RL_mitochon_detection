import torch
import torch.nn as nn
from pathlib import Path
import math
import platform
from .modules import *

class BaseModel(nn.Module):
    """
    The BaseModel class serves as a base class for all the models in the Ultralytics YOLO family.
    """

    def forward(self, x, profile=False, visualize=False):
        """
        Forward pass of the model on a single scale.
        Wrapper for `_forward_once` method.

        Args:
            x (torch.Tensor): The input image tensor
            profile (bool): Whether to profile the model, defaults to False
            visualize (bool): Whether to return the intermediate feature maps, defaults to False

        Returns:
            (torch.Tensor): The output of the network.
        """
        return self._forward_once(x, profile, visualize)

    def _forward_once(self, x, profile=False, visualize=False):
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): The input tensor to the model
            profile (bool):  Print the computation time of each layer if True, defaults to False.
            visualize (bool): Save the feature maps of the model if True, defaults to False

        Returns:
            (torch.Tensor): The last output of the model.
        """
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run
            y.append(x if m.i in self.save else None)  # save output
        return x

    def _profile_one_layer(self, m, x, dt):
        """
        Profile the computation time and FLOPs of a single layer of the model on a given input. Appends the results to the provided list.

        Args:
            m (nn.Module): The layer to be profiled.
            x (torch.Tensor): The input data to the layer.
            dt (list): A list to store the computation time of the layer.

        Returns:
            None
        """
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        for _ in range(10):
            m(x.copy() if c else x)

    def fuse(self):
        """
        Fuse the `Conv2d()` and `BatchNorm2d()` layers of the model into a single layer, in order to improve the computation efficiency.

        Returns:
            (nn.Module): The fused model is returned.
        """
        for m in self.model.modules():
            if isinstance(m, (Conv, DWConv)) and hasattr(m, 'bn'):
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                delattr(m, 'bn')  # remove batchnorm
                m.forward = m.forward_fuse  # update forward
        self.info()
        return self

    def info(self, verbose=False, imgsz=640):
        """
        Prints model information

        Args:
            verbose (bool): if True, prints out the model information. Defaults to False
            imgsz (int): the size of the image that the model will be trained on. Defaults to 640
        """
        model_info(self, verbose, imgsz)

    def _apply(self, fn):
        """
        `_apply()` is a function that applies a function to all the tensors in the model that are not
        parameters or registered buffers

        Args:
            fn: the function to apply to the model

        Returns:
            A model that is a Detect() object.
        """
        self = super()._apply(fn)
        m = self.model[-1]  # Detect()
        if isinstance(m, (Detect, Segment)):
            m.stride = fn(m.stride)
            m.anchors = fn(m.anchors)
            m.strides = fn(m.strides)
        return self

    def load(self, weights):
        """
        This function loads the weights of the model from a file

        Args:
            weights (str): The weights to load into the model.
        """
        # Force all tasks to implement this function
        raise NotImplementedError("This function needs to be implemented by derived classes!")