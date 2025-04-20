import torch 
import torch.nn as nn
class PatchEmbeddings(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    """

    def __init__(self, config):
        super().__init__()
        self.image_size = config["image_size"]
        self.patch_size = config["patch_size"]
        self.num_channels = config["num_channels"]
        self.hidden_size = config["hidden_size"]
        # Calculate the number of patches from the image size and patch size
        self.num_patches = (self.image_size // self.patch_size) ** 2
        # Create a projection layer to convert the image into patches
        # The layer projects each patch into a vector of size hidden_size
        self.projection = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)

    def forward(self, x):
        # (batch_size, num_channels, image_size, image_size) -> (batch_size, num_patches, hidden_size)
        x = self.projection(x)
        print(x.shape)
        x = x.flatten(2).transpose(1, 2)
        return x

def main():
    # config = {
    #     "image_size": 224,
    #     "patch_size": 16,
    #     "num_channels": 3,
    #     "hidden_size" : 48,
    # }
    # pe = PatchEmbeddings(config)
    # x = torch.rand((48,3,224,224))
    # output = pe(x)
    # print(output.shape)

    x = torch.rand((3,2))
    print(x)
    print(x.transpose(0,1).contiguous().view(2,3))

if __name__ == "__main__":
    main()