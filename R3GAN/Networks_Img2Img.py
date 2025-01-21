import math
import torch
import torch.nn as nn

# import sys
# sys.path.append("..")

try:
    from .Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
    from .FusedOperators import BiasedActivation
except:
    from Resamplers import InterpolativeUpsampler, InterpolativeDownsampler
    from FusedOperators import BiasedActivation


def MSRInitializer(Layer, ActivationGain=1):
    FanIn = Layer.weight.data.size(1) * Layer.weight.data[0][0].numel()
    Layer.weight.data.normal_(0, ActivationGain / math.sqrt(FanIn))

    if Layer.bias is not None:
        Layer.bias.data.zero_()

    return Layer


class Convolution(nn.Module):
    def __init__(
        self, InputChannels, OutputChannels, KernelSize, Groups=1, ActivationGain=1, padding=None
    ):
        super(Convolution, self).__init__()
        if padding is None:
            padding = (KernelSize - 1) // 2

        self.Layer = MSRInitializer(
            nn.Conv2d(
                InputChannels,
                OutputChannels,
                kernel_size=KernelSize,
                stride=1,
                padding=padding,
                groups=Groups,
                bias=False,
            ),
            ActivationGain=ActivationGain,
        )

    def forward(self, x):
        return nn.functional.conv2d(
            x,
            self.Layer.weight.to(x.dtype),
            padding=self.Layer.padding,
            groups=self.Layer.groups,
        )


class ResidualBlock(nn.Module):
    def __init__(
        self,
        InputChannels,
        Cardinality,
        ExpansionFactor,
        KernelSize,
        VarianceScalingParameter,
    ):
        super(ResidualBlock, self).__init__()

        NumberOfLinearLayers = 3
        ExpandedChannels = InputChannels * ExpansionFactor
        ActivationGain = BiasedActivation.Gain * VarianceScalingParameter ** (
            -1 / (2 * NumberOfLinearLayers - 2)
        )

        self.LinearLayer1 = Convolution(
            InputChannels, ExpandedChannels, KernelSize=1, ActivationGain=ActivationGain
        )
        self.LinearLayer2 = Convolution(
            ExpandedChannels,
            ExpandedChannels,
            KernelSize=KernelSize,
            Groups=Cardinality,
            ActivationGain=ActivationGain,
        )
        self.LinearLayer3 = Convolution(
            ExpandedChannels, InputChannels, KernelSize=1, ActivationGain=0
        )

        self.NonLinearity1 = BiasedActivation(ExpandedChannels)
        self.NonLinearity2 = BiasedActivation(ExpandedChannels)

    def forward(self, x):
        y = self.LinearLayer1(x)
        y = self.LinearLayer2(self.NonLinearity1(y))
        y = self.LinearLayer3(self.NonLinearity2(y))

        return x + y


class UpsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(UpsampleLayer, self).__init__()

        self.Resampler = InterpolativeUpsampler(ResamplingFilter)

        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)

    def forward(self, x):
        x = self.LinearLayer(x) if hasattr(self, "LinearLayer") else x
        x = self.Resampler(x)

        return x


class DownsampleLayer(nn.Module):
    def __init__(self, InputChannels, OutputChannels, ResamplingFilter):
        super(DownsampleLayer, self).__init__()

        self.Resampler = InterpolativeDownsampler(ResamplingFilter)

        if InputChannels != OutputChannels:
            self.LinearLayer = Convolution(InputChannels, OutputChannels, KernelSize=1)

    def forward(self, x):
        x = self.Resampler(x)
        x = self.LinearLayer(x) if hasattr(self, "LinearLayer") else x

        return x


class GenerativeBasis(nn.Module):
    def __init__(self, InputDimension, OutputChannels):
        super(GenerativeBasis, self).__init__()

        self.Basis = nn.Parameter(torch.empty(OutputChannels, 4, 4).normal_(0, 1))
        self.LinearLayer = MSRInitializer(
            nn.Linear(InputDimension, OutputChannels, bias=False)
        )

    def forward(self, x):
        # print("x", x.shape)
        # print("self.Basis", self.Basis.shape)
        return self.Basis.view(1, -1, 4, 4) * self.LinearLayer(x).view(
            x.shape[0], -1, 1, 1
        )


class DiscriminativeBasis(nn.Module):
    def __init__(self, InputChannels, OutputDimension):
        super(DiscriminativeBasis, self).__init__()

        self.Basis = MSRInitializer(
            nn.Conv2d(
                InputChannels,
                InputChannels,
                kernel_size=4,
                stride=1,
                padding=0,
                groups=InputChannels,
                bias=False,
            )
        )
        self.LinearLayer = MSRInitializer(
            nn.Linear(InputChannels, OutputDimension, bias=False)
        )

    def forward(self, x):
        return self.LinearLayer(self.Basis(x).view(x.shape[0], -1))


class GeneratorStage(nn.Module):
    def __init__(
        self,
        InputChannels,
        OutputChannels,
        Cardinality,
        NumberOfBlocks,
        ExpansionFactor,
        KernelSize,
        VarianceScalingParameter,
        ResamplingFilter=None,
        DataType=torch.float32,
        FuseType="concat",
    ):
        super(GeneratorStage, self).__init__()
        self.FuseType = FuseType

        TransitionLayer = (
            GenerativeBasis(InputChannels, OutputChannels)
            if ResamplingFilter is None
            else UpsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        )
        if FuseType == "concat":
            FuseBlock = Convolution(
                2 * OutputChannels, OutputChannels, KernelSize=1
            )
        elif FuseType == "add":
            FuseBlock = nn.Identity()
            
        self.Layers = nn.ModuleList(
            [TransitionLayer] + [FuseBlock]
            + [
                ResidualBlock(
                    OutputChannels,
                    Cardinality,
                    ExpansionFactor,
                    KernelSize,
                    VarianceScalingParameter,
                )
                for _ in range(NumberOfBlocks)
            ]
        )
        self.DataType = DataType

    def forward(self, x, feature):
        x = x.to(self.DataType)
        feature = feature.to(self.DataType)
        x = self.Layers[0](x)
        if self.FuseType == "concat":
            # print("x", x.shape, "feature", feature.shape); exit(0)
            x = torch.cat([x, feature], dim=1)
        elif self.FuseType == "add":
            x = x + feature
        
        for Layer in self.Layers[1:]:
            x = Layer(x)

        return x


class DiscriminatorStage(nn.Module):
    def __init__(
        self,
        InputChannels,
        OutputChannels,
        Cardinality,
        NumberOfBlocks,
        ExpansionFactor,
        KernelSize,
        VarianceScalingParameter,
        ResamplingFilter=None,
        DataType=torch.float32,
    ):
        super(DiscriminatorStage, self).__init__()

        TransitionLayer = (
            DiscriminativeBasis(InputChannels, OutputChannels)
            if ResamplingFilter is None
            else DownsampleLayer(InputChannels, OutputChannels, ResamplingFilter)
        )
        self.Layers = nn.ModuleList(
            [
                ResidualBlock(
                    InputChannels,
                    Cardinality,
                    ExpansionFactor,
                    KernelSize,
                    VarianceScalingParameter,
                )
                for _ in range(NumberOfBlocks)
            ]
            + [TransitionLayer]
        )
        self.DataType = DataType

    def forward(self, x):
        x = x.to(self.DataType)

        for Layer in self.Layers:
            x = Layer(x)

        return x


class Generator(nn.Module):
    def __init__(
        self,
        NoiseDimension,
        WidthPerStage,
        CardinalityPerStage,
        BlocksPerStage,
        ExpansionFactor,
        ConditionDimension=None,
        ConditionEmbeddingDimension=0,
        KernelSize=3,
        ResamplingFilter=[1, 2, 1],
        FuseType="concat",
    ):
        super(Generator, self).__init__()
        self.FuseType = FuseType

        VarianceScalingParameter = sum(BlocksPerStage)
        MainLayers = [
            GeneratorStage(
                NoiseDimension + ConditionEmbeddingDimension,
                WidthPerStage[0],
                CardinalityPerStage[0],
                BlocksPerStage[0],
                ExpansionFactor,
                KernelSize,
                VarianceScalingParameter,
            )
        ]
        MainLayers += [
            GeneratorStage(
                WidthPerStage[x],
                WidthPerStage[x + 1],
                CardinalityPerStage[x + 1],
                BlocksPerStage[x + 1],
                ExpansionFactor,
                KernelSize,
                VarianceScalingParameter,
                ResamplingFilter,
            )
            for x in range(len(WidthPerStage) - 1)
        ]

        self.MainLayers = nn.ModuleList(MainLayers)
        self.AggregationLayer = Convolution(WidthPerStage[-1], 3, KernelSize=1)

    def forward(self, x, features=None, y=None):
        # 第一步，先常规根据 latent 生成

        for Layer, feature in zip(self.MainLayers, features):
            x = Layer(x, feature)

        return self.AggregationLayer(x)


class Image2Image(nn.Module):
    def __init__(
        self,
        InputDimension,
        NoiseDimension,
        WidthPerStage,
        CardinalityPerStage,
        BlocksPerStage,
        ExpansionFactor,
        ConditionDimension=None,
        ConditionEmbeddingDimension=0,
        KernelSize=3,
        ResamplingFilter=[1, 2, 1],
        FuseType="concat",
    ):
        super(Image2Image, self).__init__()
        WidthPerStage.reverse()
        CardinalityPerStage.reverse()
        BlocksPerStage.reverse()
        VarianceScalingParameter = sum(BlocksPerStage)
        EncodeLayers = [Convolution(InputDimension, WidthPerStage[0], KernelSize=1)]
        EncodeLayers += [
            DiscriminatorStage(
                WidthPerStage[x],
                WidthPerStage[x + 1],
                CardinalityPerStage[x],
                BlocksPerStage[x],
                ExpansionFactor,
                KernelSize,
                VarianceScalingParameter,
                ResamplingFilter,
            )
            for x in range(len(WidthPerStage) - 1)
        ]
        self.EncodeLayers = nn.ModuleList(EncodeLayers)

        assert FuseType in ["concat", "add"]

        self.EncFinalLayer = Convolution(
            WidthPerStage[-1], NoiseDimension, KernelSize=4, padding=0
        )

        # Decoder / Generator
        WidthPerStage.reverse()
        CardinalityPerStage.reverse()
        BlocksPerStage.reverse()
        self.generator = Generator(
            NoiseDimension,
            WidthPerStage,
            CardinalityPerStage,
            BlocksPerStage,
            ExpansionFactor,
            ConditionDimension,
            ConditionEmbeddingDimension,
            KernelSize,
            ResamplingFilter,
            FuseType,
        )

    def forward(self, x):
        # x: (batch, 3, 256, 256)
        features = []
        # (batch, 96, 256, 256)
        # (batch, 192, 128, 128)
        # (batch, 384, 64, 64)
        # (batch, 768, 32, 32)
        # (batch, 768, 16, 16)
        # (batch, 768, 8, 8)
        # (batch, 768, 4, 4)
        for Layer in self.EncodeLayers:
            x = Layer(x)
            features.append(x)

        latent = self.EncFinalLayer(x).squeeze(-1).squeeze(-1)
        # print("latent", latent.shape, "x", x.shape); exit(0)
        out = self.generator(latent, features[::-1])
        return out


class Discriminator(nn.Module):
    def __init__(
        self,
        WidthPerStage,
        CardinalityPerStage,
        BlocksPerStage,
        ExpansionFactor,
        ConditionDimension=None,
        ConditionEmbeddingDimension=0,
        KernelSize=3,
        ResamplingFilter=[1, 2, 1],
    ):
        super(Discriminator, self).__init__()

        VarianceScalingParameter = sum(BlocksPerStage)
        MainLayers = [
            DiscriminatorStage(
                WidthPerStage[x],
                WidthPerStage[x + 1],
                CardinalityPerStage[x],
                BlocksPerStage[x],
                ExpansionFactor,
                KernelSize,
                VarianceScalingParameter,
                ResamplingFilter,
            )
            for x in range(len(WidthPerStage) - 1)
        ]
        MainLayers += [
            DiscriminatorStage(
                WidthPerStage[-1],
                1 if ConditionDimension is None else ConditionEmbeddingDimension,
                CardinalityPerStage[-1],
                BlocksPerStage[-1],
                ExpansionFactor,
                KernelSize,
                VarianceScalingParameter,
            )
        ]

        self.ExtractionLayer = Convolution(3, WidthPerStage[0], KernelSize=1)
        self.MainLayers = nn.ModuleList(MainLayers)

        if ConditionDimension is not None:
            self.EmbeddingLayer = MSRInitializer(
                nn.Linear(ConditionDimension, ConditionEmbeddingDimension, bias=False),
                ActivationGain=1 / math.sqrt(ConditionEmbeddingDimension),
            )

    def forward(self, x, y=None):
        x = self.ExtractionLayer(x.to(self.MainLayers[0].DataType))

        for Layer in self.MainLayers:
            x = Layer(x)

        x = (
            (x * self.EmbeddingLayer(y)).sum(dim=1, keepdim=True)
            if hasattr(self, "EmbeddingLayer")
            else x
        )

        return x.view(x.shape[0])

def get_256_model(FuseType="concat"):
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
        FuseType=FuseType,
    )
    return model

def get_512_model(FuseType="concat"):
    model = Image2Image(
        InputDimension=3,
        NoiseDimension=768,
        WidthPerStage=[768, 768, 768, 768, 768, 384, 192, 96],
        CardinalityPerStage=[96, 96, 96, 96, 96, 96, 96, 96],
        BlocksPerStage=[2, 2, 2, 2, 2, 2, 2, 2],
        ExpansionFactor=2,
        ConditionDimension=None,
        ConditionEmbeddingDimension=0,
        KernelSize=3,
        ResamplingFilter=[1, 2, 1],
        FuseType=FuseType,
    )
    return model

if __name__ == "__main__":
    device = torch.device("cuda:7")
    model = get_256_model.to(device)
    inp = torch.randn(1, 3, 256, 256).to(device)
    out = model(inp)
    print(out.shape)
