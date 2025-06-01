## What is this?

A script that converts torch pretrained weights for vgg-19 arch to be used with vgg-19 implemented in Jax&Equinox


## Special appearance:

`tree_flatten` and `tree_unflatten` from `jax.treeutils` AND `partition` and `combine` from equinox for pytree splendor

## Why bother?

1. The need to migrate away from anything pytorch and work exclusively with jax&equinox
2. Making pytorch and jax work side by side is a dependency nightmare (been there, done that and no thank you)
