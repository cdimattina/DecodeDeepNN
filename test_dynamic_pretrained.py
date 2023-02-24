
from dynamic_pretrained_model import DynamicPretrainedModel

dynamic = DynamicPretrainedModel("Xception", 1, (60, 60, 3), 1)
dynamic.model_summary()
