from transformers import AutoModelForQuestionAnswering
from transformers import AutoTokenizer
import config

def load_model(model_name = None):
    """
    Loads a pre-trained model for question-answering tasks.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        model: The loaded model.
    """
    if model_name is None:
        model_name = config.MODEL_NAME

    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return model

def get_tokenizer(model_name = None):
    """
    Loads the tokenizer for the specified model.

    Args:
        model_name (str): The name of the pre-trained model to load.

    Returns:
        tokenizer: The loaded tokenizer.
    """
    if model_name is None:
        model_name = config.MODEL_NAME

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer
