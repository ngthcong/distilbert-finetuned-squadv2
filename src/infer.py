import config
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def validate_input(text):
    """
    Validates the input text for the model.

    Args:
        text (str): The input text to validate.

    Returns:
        tuple: A tuple containing a boolean indicating validity and an error message (if any).
    """
    if not text.strip():
        return False, "Input is empty."
    if any(ord(c) < 32 and c not in '\n\t' for c in text):
        return False, "Input contains invalid control characters."
    return True, ""

def query_model( context, question):
    """
    Queries the model with a given context and question.

    Args:
        context (str): The context text.
        question (str): The question text.

    Returns:
        str: The answer predicted by the model.
    """
    is_valid, error_message = validate_input(context)
    if not is_valid:
        raise ValueError(error_message)

    is_valid, error_message = validate_input(question)
    if not is_valid:
        raise ValueError(error_message)

    try:
        pipe = load_model_pipeline()
        result = pipe(context=context, question=question)
        return result['answer']
    except Exception as e:
        raise RuntimeError(f"Error querying model: {e}")

def load_model_pipeline():
    """
    Loads the model pipeline for question-answering.

    Returns:
        pipeline: The loaded pipeline for question-answering.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        model = AutoModelForQuestionAnswering.from_pretrained(config.MODEL_NAME)
        pipe = pipeline ( config.PIPELINE_NAME , model = model ,tokenizer=tokenizer)
        return pipe
    except Exception as e:
        raise RuntimeError(f"Error loading pipeline: {e}")
    
if __name__ == "__main__":
    import os
    context = "My name is AI Vietnam and I live in Vietnam."
    question = "Where do I live?"
    try:
        print("Answer:", query_model(context, question))
    except Exception as e:
        print(f"Error: {e}")