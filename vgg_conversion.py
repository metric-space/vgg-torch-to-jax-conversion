import os
import requests
from PIL import Image
from io import BytesIO

import equinox as eqx

import jax
import jax.random as jr
import jax.numpy as jnp

import torchvision.models as models


class VGGBlock(eqx.Module):

    layers: eqx.nn.Sequential
    maxpool: eqx.nn.MaxPool2d = eqx.static_field()

    def __init__(self, key, in_channels, out_channels, num_blocks):

        keys = jr.split(key, num_blocks)

        self.layers = []

        for i in range(num_blocks):
            self.layers.extend(
                [
                    eqx.nn.Conv2d(
                        key=keys[i],
                        padding=1,
                        kernel_size=3,
                        in_channels=in_channels if i == 0 else out_channels,
                        out_channels=out_channels,
                    ),
                    eqx.nn.Lambda(jax.nn.relu),
                ]
            )

        self.layers = eqx.nn.Sequential(self.layers)

        self.maxpool = eqx.nn.MaxPool2d(kernel_size=2, stride=2)

    def __call__(self, x):
        x = self.layers(x)
        x = self.maxpool(x)
        return x


class MLPHead(eqx.Module):

    classifier: eqx.nn.Sequential

    def __init__(self, key):

        keys = jr.split(key, 3)

        self.classifier = eqx.nn.Sequential(
            [
                eqx.nn.Linear(512 * 7 * 7, 4096, key=keys[0]),
                eqx.nn.Linear(4096, 4096, key=keys[1]),
                eqx.nn.Linear(4096, 1000, key=keys[2]),
            ]
        )

    def __call__(self, x):

        print(x.shape)

        return self.classifier(x.reshape(-1))


class VGG19(eqx.Module):

    layers: eqx.nn.Sequential
    classifier: MLPHead

    def __init__(self, key):

        block_confs = [
            (3, 64, 2),
            (64, 128, 2),
            (128, 256, 4),
            (256, 512, 4),
            (512, 512, 4),
        ]

        keys = jr.split(key, len(block_confs) + 1)  # 1 for the classifier

        self.layers = eqx.nn.Sequential(
            [
                eqx.nn.Lambda(VGGBlock(keys[i], *conf))
                for i, conf in enumerate(block_confs)
            ]
        )

        self.classifier = MLPHead(key=keys[-1])

    def __call__(self, x):

        return self.classifier(self.layers(x))


def image_preprocessing_pipeline(image):

    image = image.resize((224, 224))
    img_jax = jnp.array(image).astype(jnp.float32) / 255.0  # [H, W, C], range [0,1]

    mean = jnp.array([0.485, 0.456, 0.406])
    std = jnp.array([0.229, 0.224, 0.225])
    img_jax = (img_jax - mean) / std

    img_jax = img_jax.transpose(2, 0, 1)

    return img_jax


def save_model(model: eqx.Module, filename: str, directory: str = "./models"):

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, filename), "wb") as f:
        eqx.tree_serialise_leaves(f, model)


if __name__ == "__main__":

    key = jax.random.PRNGKey(2345)

    vgg19 = models.vgg19(pretrained=True)

    jax_model = VGG19(key)

    model_tree_to_replace, model_tree_not_to_replace = eqx.partition(
        jax_model, eqx.is_array
    )

    leaves, treedef = jax.tree_util.tree_flatten(model_tree_to_replace)

    pt_params = []
    for i, (k, v) in enumerate(vgg19.state_dict().items()):
        arr = v.numpy()
        if arr.ndim != leaves[i].ndim:
            arr = jnp.array(arr)[:, None, None]
        else:
            arr = jnp.array(arr)
        pt_params.append(arr)

    for i, j in list(
        zip([leaf.shape for leaf in leaves], [param.shape for param in pt_params])
    ):
        assert i == j

    revitalized_jax_model = treedef.unflatten(pt_params)
    revitalized_jax_model = eqx.combine(
        revitalized_jax_model, model_tree_not_to_replace
    )

    url = "https://upload.wikimedia.org/wikipedia/commons/9/9a/Pug_600.jpg"
    image = Image.open(BytesIO(requests.get(url).content)).convert("RGB")

    img_jax = image_preprocessing_pipeline(image)

    logits = revitalized_jax_model(img_jax)
    top1 = jnp.argmax(logits, axis=-1)

    imagenet_labels = requests.get(
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
    ).text.splitlines()

    print(f"Top-1 prediction: {imagenet_labels[int(top1)]}")

    print("Saving model")
    save_model(revitalized_jax_model, "vgg19.eqx")
