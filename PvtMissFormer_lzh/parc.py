import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
from torch import nn


class ParCUnit(torch.nn.Module):
    # Image_height, image_width -> size (int or tuple)
    # In_channels==out_channels
    # Fast Parc won't work with in_channels != out_channels
    # And it will be able to perform only depthwise convolution
    def __init__(self, interpolation_points, inner_channels,
                 image_size,
                 interpolation_type="bilinear", orientation="H", 
                 depthwise = False): # Depthwise -> True
        super().__init__()

        self.orientation = orientation
        self.interpolation_type = interpolation_type
        image_height, image_width = image_size

        in_channels, out_channels = inner_channels, inner_channels

        width_size, height_size = 1, 1
        self.groups = 1
        if depthwise:
            self.groups = in_channels
            in_channels = 1
        if orientation == "H":
            width_size = interpolation_points
            image_height = 1
        if orientation == "V":
            height_size = interpolation_points
            image_width = 1
        positional_codes_values = torch.rand(image_height, image_width)

        weights = torch.rand((out_channels, in_channels, height_size, width_size))
        weights = torch.nn.parameter.Parameter(weights)
        self.weights = weights

        bias_tensor = torch.rand(out_channels)
        self.bias_parameters = torch.nn.parameter.Parameter(bias_tensor)

        self.positional_encoding = torch.nn.parameter.Parameter(positional_codes_values)

    def apply_convoution(self, X):
        target_width, target_height = 1, 1
        if self.orientation == "H":
            target_width = X.shape[2]
        if self.orientation == "V":
            target_height = X.shape[3]

        conv_parameters = None
        conv_parameters = F.interpolate(self.weights, size = (target_height, target_width),
                                        mode = self.interpolation_type)
        
        output = conv2d(X, weight = conv_parameters, bias = self.bias_parameters, groups = self.groups)
        output += self.positional_encoding
        return output
        
    
    def forward(self, X):
        if self.orientation == "H":
            X_cat = torch.cat((X, X[:, :, :, :-1]), dim=-1)
        if self.orientation == "V":
            X_cat = torch.cat((X, X[:, :, :-1, :]), dim=-2)
        return self.apply_convoution(X_cat)


class ParCBlock(nn.Module):
    def __init__(self, interpolation_points, inner_channels,
                 image_size, # image_height x image_width
                 interpolation_type = "bilinear", depthwise = False):
        super().__init__()
                
        self.parc_h = ParCUnit(interpolation_points=interpolation_points, orientation="H",
            inner_channels=inner_channels//2, 
            image_size=image_size,
            interpolation_type=interpolation_type, depthwise=depthwise)
                
        self.parc_v = ParCUnit(interpolation_points=interpolation_points, orientation="V",
            inner_channels=inner_channels//2, 
            image_size=image_size,
            interpolation_type=interpolation_type, depthwise=depthwise)
        
        
    def forward(self, input):
        channels = input.shape[1]
        input_h = input[:, :channels//2, :, :]
        input_v = input[:, channels//2:, :, :]
        output_h = self.parc_h(input_h)
        output_v = self.parc_v(input_v)
        return torch.cat((output_h, output_v), dim=1)


class ParCNextNeck(nn.Module):
    def __init__(self, input_channels, hidden_channels, out_channels, image_size,
                 interpolation_points=10, fast=False):
        super().__init__()
        self.layernorm = nn.LayerNorm(normalized_shape=image_size)
        if not fast:
            self.parc_block = ParCBlock(interpolation_points, input_channels,
                                        image_size, depthwise=True)

        self.bottleneck_extender = nn.Conv2d(input_channels, hidden_channels, kernel_size=1)
        self.bottleneck_reductor = nn.Conv2d(hidden_channels, input_channels, kernel_size=1)

        self.extend_channels_if_necessary = nn.Identity()
        if input_channels != out_channels:
            self.extend_channels_if_necessary = nn.Conv2d(in_channels=input_channels, out_channels=out_channels,
                                                          kernel_size=1)

        self.projector = nn.Identity()
        self.maxpool_if_necessary = nn.Identity()
        if input_channels != out_channels:
            self.projector = nn.Conv2d(in_channels=input_channels, out_channels=out_channels,
                                       kernel_size=1, stride=2)
            self.maxpool_if_necessary = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, input):
        x = self.layernorm(input)
        x = self.parc_block(x)
        x = self.bottleneck_extender(x)
        x = nn.GELU()(x)
        x = self.bottleneck_reductor(x)
        x = self.extend_channels_if_necessary(x)
        x = self.maxpool_if_necessary(x)
        return x + self.projector(input)


if __name__ == '__main__':
    x = torch.randn(24, 512, 7, 7)
    # reduction_ratios = [1, 2, 4, 8]
    model = ParCNextNeck(512, 64, 512, [7, 7])
    out = model(x)
    print("out", out.shape)