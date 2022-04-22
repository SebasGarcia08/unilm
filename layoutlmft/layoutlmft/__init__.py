from collections import OrderedDict

from transformers import CONFIG_MAPPING, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, MODEL_NAMES_MAPPING, TOKENIZER_MAPPING
from transformers.convert_slow_tokenizer import SLOW_TO_FAST_CONVERTERS, BertConverter, XLMRobertaConverter

# Removed auto_class_factory import because was not found in current speicifed transformers version
# See: https://github.com/huggingface/transformers/issues/13591
from transformers.models.auto.modeling_auto import _BaseAutoModelClass, auto_class_update
import types

from .models.layoutlmv2 import (
    LayoutLMv2Config,
    LayoutLMv2ForRelationExtraction,
    LayoutLMv2ForTokenClassification,
    LayoutLMv2Tokenizer,
    LayoutLMv2TokenizerFast,
)
from .models.layoutxlm import (
    LayoutXLMConfig,
    LayoutXLMForRelationExtraction,
    LayoutXLMForTokenClassification,
    LayoutXLMTokenizer,
    LayoutXLMTokenizerFast,
)


CONFIG_MAPPING.update([("layoutlmv2", LayoutLMv2Config), ("layoutxlm", LayoutXLMConfig)])
MODEL_NAMES_MAPPING.update([("layoutlmv2", "LayoutLMv2"), ("layoutxlm", "LayoutXLM")])
TOKENIZER_MAPPING.update(
    [
        (LayoutLMv2Config, (LayoutLMv2Tokenizer, LayoutLMv2TokenizerFast)),
        (LayoutXLMConfig, (LayoutXLMTokenizer, LayoutXLMTokenizerFast)),
    ]
)
SLOW_TO_FAST_CONVERTERS.update({"LayoutLMv2Tokenizer": BertConverter, "LayoutXLMTokenizer": XLMRobertaConverter})
MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING.update(
    [(LayoutLMv2Config, LayoutLMv2ForTokenClassification), (LayoutXLMConfig, LayoutXLMForTokenClassification)]
)

MODEL_FOR_RELATION_EXTRACTION_MAPPING = OrderedDict(
    [(LayoutLMv2Config, LayoutLMv2ForRelationExtraction), (LayoutXLMConfig, LayoutXLMForRelationExtraction)]
)


# Added these try-except blocks because were causing problems with custom models specified here
# See: https://github.com/huggingface/transformers/issues/13591
try:
    AutoModelForTokenClassification = auto_class_update(
        "AutoModelForTokenClassification", MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING, head_doc="token classification"
    )
except:
    cls = types.new_class("AutoModelForTokenClassification", (_BaseAutoModelClass,))
    cls._model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
    cls.name = "AutoModelForTokenClassification"
    AutoModelForTokenClassification = auto_class_update(cls, head_doc="token classification")


try:
    AutoModelForRelationExtraction = auto_class_update(
        "AutoModelForRelationExtraction", MODEL_FOR_RELATION_EXTRACTION_MAPPING, head_doc="relation extraction"
    )
except:
    cls = types.new_class("AutoModelForRelationExtraction", (_BaseAutoModelClass,))
    cls._model_mapping = MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING
    cls.name = "AutoModelForRelationExtraction"
    AutoModelForRelationExtraction = auto_class_update(cls, head_doc="relation extraction")
