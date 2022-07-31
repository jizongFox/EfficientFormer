from typing import Type

from torch import nn

from models.efficientformer import efficientformer_l1


def convert_from_to(model, *, from_: Type[nn.Module], to_: Type[nn.Module]):
    for child_name, child in model.named_children():
        if isinstance(child, from_):
            setattr(model, child_name, to_())
        else:
            convert_from_to(child, from_=from_, to_=to_)


if __name__ == '__main__':
    model = efficientformer_l1()
    print(model)
    from loguru import logger

    logger.info("converting Gelu to Relu")

    convert_from_to(model, from_=nn.GELU, to_=nn.ReLU)
    print(model)
