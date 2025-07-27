import collections
import numpy as np
from tqdm import tqdm
from config import N_BEST, MAX_ANS_LENGTH
import evaluate

def compute_metrics(start_logits, end_logits, features, examples):
    """
    Computes evaluation metrics for a question-answering task.

    This function processes the model's start and end logits, maps them to 
    token positions, and generates predicted answers. It compares the 
    predictions with the ground truth answers to compute evaluation metrics.

    Args:
        start_logits (list of list of float): Start logits for each feature.
        end_logits (list of list of float): End logits for each feature.
        features (list of dict): List of features containing tokenized inputs 
            and offset mappings.
        examples (list of dict): List of examples containing:
            - "id" (str): Example ID.
            - "context" (str): Context text.
            - "answers" (dict): Ground truth answers with:
                - "text" (list of str): List of answer texts.
                - "answer_start" (list of int): List of start character positions.

    Returns:
        dict: Evaluation metrics computed using the `metric` object.
    """
    metric = eval.load("squad_v2")
    # Create a default dictionary to map each example
    # to its corresponding list of features
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature['example_id']].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example['id']
        context = example['context']
        answers = []
        # Iterate through all features related to this example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]['offset_mapping']
            # Get the indices with the highest values for start and end logits
            start_indexes = np.argsort(start_logit)[-1: -N_BEST - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -N_BEST - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully within the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers longer than max_answer_length
                    if end_index - start_index + 1 > MAX_ANS_LENGTH:
                        continue

                    # Create a new answer
                    text = context[offsets[start_index][0]: offsets[end_index][1]]
                    logit_score = start_logit[start_index] + end_logit[end_index]
                    answer = {
                        'text': text,
                        'logit_score': logit_score,
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x['logit_score'])

            answer_dict = {
                'id': example_id,
                'prediction_text': best_answer['text'],
                'no_answer_probability': 1 - best_answer['logit_score']
            }
        else:
            answer_dict = {
                'id': example_id,
                'prediction_text': '',
                'no_answer_probability': 1.0
            }
        predicted_answers.append(answer_dict)

    # Create a list of theoretical answers from the examples
    theoretical_answers = [
        {'id': ex['id'], 'answers': ex['answers']} for ex in examples
    ]
    # Use metric.compute to calculate metrics and return the results
    return metric.compute(
        predictions=predicted_answers,
        references=theoretical_answers
    )