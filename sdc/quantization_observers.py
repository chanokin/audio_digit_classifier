### from https://discuss.pytorch.org/t/custom-weight-observer-for-powers-of-2/191749
import torch
from torch.quantization import MinMaxObserver
import numpy as np

class PowerOfTwoMinMaxObserver(MinMaxObserver):
    """
    Observer module for computing the quantization parameters based on the
    running min and max values, with scales as powers of two.

    This observer extends the MinMaxObserver to use scales that are powers of two.
    It overrides the calculate_qparams method to compute the power of two scale.
    """

    def calculate_qparams(self):
        r"""Calculates the quantization parameters with scale as a power of two."""
        min_val, max_val = self.min_val.item(), self.max_val.item()

        # Calculate the scale as the nearest power of two
        max_range = max(abs(min_val), abs(max_val))
        scale = 2 ** np.ceil(np.log2(max_range / (self.quant_max - self.quant_min)))

        # print(f"{scale} {max_range} {min_val} {max_val} {self.quant_max} {self.quant_min}")

        # Calculate zero_point as in the base class
        if self.qscheme == torch.per_tensor_symmetric:
            if self.dtype == torch.qint8:
                zero_point = 0
            else:
                zero_point = 128
        else:
            zero_point = self.quant_min - round(min_val / scale)

        # Convert scale and zero_point to PyTorch tensors
        scale = torch.tensor(scale, dtype=torch.float32)
        zero_point = torch.tensor(zero_point, dtype=torch.int64)
        return scale, zero_point

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}, scale=power of two"

# qconfig = quantization.get_default_qat_qconfig('fbgemm')
# custom_activation_observer = PowerOfTwoMinMaxObserver.with_args()  # Create an instance of your custom observer

# # Set the custom observer for activations in the qconfig
# qconfig = torch.quantization.default_qconfig._replace(activation=custom_activation_observer)

class PowerOfTwoWeightObserver(MinMaxObserver):
    """
    Observer module for computing the quantization parameters based on the
    running min and max values, with scales as powers of two for weights.

    This observer extends the MinMaxObserver to use scales that are powers of two.
    It overrides the calculate_qparams method to compute the power of two scale.
    """

    def __init__(self, bit_width=8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dtype = torch.qint8  # Default dtype
        self.bit_width = bit_width  # Specify the bit width

    def calculate_qparams(self):
        r"""Calculates the quantization parameters with scale as a power of two."""
        min_val, max_val = self.min_val.item(), self.max_val.item()

        # Calculate the scale as the nearest power of two
        max_range = max(abs(min_val), abs(max_val))
        scale = 2 ** np.ceil(np.log2(max_range / (self.quant_max - self.quant_min)))

        # Calculate zero_point as in the base class
        if self.qscheme == torch.per_tensor_symmetric:
            zero_point = 0
        else:
            zero_point = self.quant_min - round(min_val / scale)

        # Convert scale and zero_point to PyTorch tensors
        scale = torch.tensor(scale, dtype=torch.float32)
        zero_point = torch.tensor(zero_point, dtype=torch.int64)

        # Adjust the scale based on the specified bit width
        scale = scale / (2 ** (self.bit_width - 1))
        return scale, zero_point

    def extra_repr(self):
        return f"min_val={self.min_val}, max_val={self.max_val}, scale=power of two, bit_width={self.bit_width}"

# custom_weight_observer = PowerOfTwoWeightObserver.with_args()
# # Set the custom observer for activations in the qconfig
# qconfig = torch.quantization.default_qconfig._replace(activation=custom_activation_observer,weight=custom_weight_observer)

