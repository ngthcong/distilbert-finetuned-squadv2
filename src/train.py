from logging import config
from transformers import TrainingArguments, Trainer
from data   import preprocess_training_examples, preprocess_validation_examples
from model  import load_model, get_tokenizer
import config
from eval import compute_metrics
from datasets import load_dataset
import logging
logging.basicConfig(level=logging.INFO)


def main():
    """
    Main function to train and evaluate a question-answering model.

    This function loads the model and tokenizer, preprocesses the training 
    and validation datasets, trains the model using the Trainer API, and 
    evaluates the model on the validation dataset.

    Args:
        None

    Returns:
        None
    """
    MODEL_NAME = "distilbert/distilbert-base-uncased"
    DATASET_NAME ='squad_v2'

    tokenizer = get_tokenizer(MODEL_NAME)

    model = load_model(MODEL_NAME)

    training_args = TrainingArguments(
    output_dir =config.OUTPUT_DIR, # Directory to save output
    evaluation_strategy ="no", # No automatic evaluation after each epoch
    save_strategy ="epoch", # Save checkpoint after each epoch
    learning_rate =2e-5, # Learning rate
    num_train_epochs =3, # Number of training epochs
    weight_decay =0.01, # Weight decay to prevent overfitting
    fp16 =True , # Use half-precision data type to optimize resources
    per_device_train_batch_size = 8, # Batch size per device for training
    )
    # Load and preprocess data
    raw_datasets = load_dataset(DATASET_NAME)

    train_dataset = raw_datasets["train"].map(
        preprocess_training_examples ,
        batched =True ,
        remove_columns = raw_datasets["train"].column_names ,
    )
    validation_dataset = raw_datasets["validation"].map(
        preprocess_validation_examples ,
        batched =True ,
        remove_columns = raw_datasets["validation"].column_names ,
    )
    trainer = Trainer (
        model =model , # Use the preloaded model
        args =training_args , # Training parameters and configurations
        train_dataset = train_dataset, # Training dataset
        eval_dataset = validation_dataset, # Validation dataset
        tokenizer = tokenizer, # Tokenizer for text processing
    )
    trainer.train()
    trainer.save_model(config.OUTPUT_DIR+ "/final_model")  # Save the trained model
    # Perform predictions on the validation dataset
    predictions , _, _ = trainer.predict(validation_dataset)

    # Extract start and end logits for predicted answers
    start_logits, end_logits = predictions

    # Compute evaluation metrics using the compute_metrics function
    results = compute_metrics(
        start_logits ,
        end_logits ,
        validation_dataset ,
        raw_datasets["validation"]
    )
    logging.info(f"Evaluation results: {results}")


if __name__ == "__main__":
    main()