import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    """
    A module to perform two consecutive convolution operations followed by batch normalization and ReLU activation.

    Attributes:
        conv (nn.Sequential): A sequential container of two convolutional blocks.

    Parameters:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Defines the computation performed at every call of the DoubleConv module.

        Parameters:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output data after passing through the convolution blocks.
        """
        return self.conv(x)

class UNET(nn.Module):
    """
    U-Net architecture for image segmentation tasks.

    Attributes:
        ups (nn.ModuleList): List of modules used in the decoder path of U-Net.
        downs (nn.ModuleList): List of modules used in the encoder path of U-Net.
        pool (nn.MaxPool2d): Max pooling layer.
        bottleneck (DoubleConv): The bottleneck layer of U-Net.
        final_conv (nn.Conv2d): Final convolutional layer to produce the output segmentation map.

    Parameters:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels in the output image.
        features (List[int]): Number of features in each layer of the network.
    """
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Defines the forward pass of the U-Net using skip connections and up-sampling.

        Parameters:
            x (torch.Tensor): The input tensor for the U-Net model.

        Returns:
            torch.Tensor: The output tensor after processing through U-Net.
        """
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)

def test():
    """
    Function to test the U-Net model to ensure it outputs the correct tensor shape.
    """
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=1)
    preds = model(x)
    assert preds.shape == x.shape

if __name__ == "__main__":
    test()
