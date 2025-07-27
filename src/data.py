from transformers import AutoTokenizer
from config import MODEL_NAME, MAX_LENGTH, STRIDE
from model import get_tokenizer

def preprocess_training_examples(examples):
    """
    Preprocesses training examples for a question-answering task.

    This function takes a dataset of examples containing questions, contexts, 
    and answers, and prepares them for training a model. It tokenizes the 
    questions and contexts, maps the answers to token positions, and returns 
    the processed inputs with start and end positions of the answers.

    Args:
        examples (dict): A dictionary containing the following keys:
            - "question" (list of str): List of questions.
            - "context" (list of str): List of contexts corresponding to the questions.
            - "answers" (list of dict): List of dictionaries containing:
                - "text" (list of str): List of answer texts.
                - "answer_start" (list of int): List of start character positions of answers.

    Returns:
        dict: A dictionary containing tokenized inputs with added keys:
            - "start_positions" (list of int): Start token positions of answers.
            - "end_positions" (list of int): End token positions of answers.
    """
    # Extract the list of questions from examples and
    # remove any extra whitespace
    questions = [q.strip() for q in examples ["question"]]

    tokenizer = get_tokenizer(MODEL_NAME)
    # Perform input encoding using the tokenizer
    inputs = tokenizer (
        questions,
        examples["context"],
        max_length = MAX_LENGTH ,
        truncation ="only_second",
        stride =STRIDE ,
        return_overflowing_tokens =True ,
        return_offsets_mapping =True ,
        padding ="max_length",
    )
    # Extract offset_mapping from inputs and remove it from inputs
    offset_mapping = inputs.pop("offset_mapping")
    
    # Extract sample_map from inputs and remove it
    sample_map = inputs.pop("overflow_to_sample_mapping")
    
    # Extract answer information from examples
    answers = examples ["answers"]
    
    # Initialize lists for start and end positions of answers
    start_positions = []
    end_positions = []
    
    # Iterate through the offset_mapping list
    for i, offset in enumerate (offset_mapping):
        # Determine the index of the sample related to the current offset
        sample_idx = sample_map[i]
    
        # Extract sequence_ids from inputs
        sequence_ids = inputs . sequence_ids (i)
    
        # Determine the start and end positions of the context
        idx = 0
        while sequence_ids [ idx ] != 1:
            idx += 1
        context_start = idx
        while sequence_ids [ idx ] == 1:
            idx += 1
        context_end = idx - 1
    
        # Extract answer information for this sample
        answer = answers [ sample_idx ]
    
        if len (answer['text']) == 0:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Determine the start and end character positions of the answer
            # in the context
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])

            # If the answer is not fully within the context,
            # assign labels as (0, 0)
            if offset [ context_start ][0] > start_char or offset [ context_end ][1] < end_char :
                start_positions . append (0)
                end_positions . append (0)
            else:
                # Otherwise, assign start and end positions based on
                # token positions
                idx = context_start
                while idx <= context_end and offset [ idx ][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)
        
                idx = context_end
                while idx >= context_start and offset [ idx ][1] >= end_char:
                    idx -= 1
                end_positions . append ( idx + 1)
    
    # Add start and end position information to inputs
    inputs ["start_positions"] = start_positions
    inputs ["end_positions"] = end_positions
    
    return inputs


def preprocess_validation_examples(examples):
    """
    Preprocesses validation examples for a question-answering task.

    This function tokenizes the questions and contexts, adjusts offset mappings,
    and prepares the inputs for validation.

    Args:
        examples (dict): A dictionary containing the following keys:
            - "question" (list of str): List of questions.
            - "context" (list of str): List of contexts corresponding to the questions.
            - "id" (list of str): List of example IDs.

    Returns:
        dict: A dictionary containing tokenized inputs with adjusted offset mappings.
    """
    # Prepare the list of questions by removing leading and trailing whitespace
    questions = [q.strip() for q in examples["question"]]

    # Use the tokenizer to encode the questions and related text
    tokenizer = get_tokenizer(MODEL_NAME)
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=MAX_LENGTH,
        truncation="only_second",
        stride=STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map each input row back to its reference example
    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    # Determine the reference example for each input row and adjust offset mappings
    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]

        # Remove offsets that do not correspond to the context
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    # Add reference example information to the inputs
    inputs["example_id"] = example_ids

    return inputs