import torch
import torch.nn as nn

from R3GAN.Networks_Img2Img import Image2Image, Discriminator


if __name__ == "__main__":
    device = torch.device("cuda:7")
    model = Image2Image(
        InputDimension=3,
        NoiseDimension=768,
        WidthPerStage=[768, 768, 768, 768, 384, 192, 96],
        CardinalityPerStage=[96, 96, 96, 96, 96, 96, 96],
        BlocksPerStage=[2, 2, 2, 2, 2, 2, 2],
        ExpansionFactor=2,
        ConditionDimension=None,
        ConditionEmbeddingDimension=0,
        KernelSize=3,
        ResamplingFilter=[1, 2, 1],
        # FuseType="concat",
        FuseType="add",
    ).to(device)
    inp = torch.randn(1, 3, 256, 256).to(device)
    out = model(inp)
    print(out.shape)


