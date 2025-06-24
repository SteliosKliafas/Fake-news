import logging

from transformers import AutoTokenizer, set_seed

from model_checkpoints import model_checkpoints_dir
from src.ml.named_entity_recognition.config.config import Config
from src.ml.named_entity_recognition.data import NERDataset, create_dataloaders, preprocess_dataset
from src.ml.named_entity_recognition.processing.training import train_model

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def ner_training_service(
    language,
    dataset,
    num_epochs=5,
):
    """
    Main function for executing Named Entity Recognition (NER) model training and evaluation.
    This function manages two modes: 'train' and 'test'. Based on the mode, it either trains the model
    on the specified dataset or loads and evaluates a pre-trained model.

    Args:
        mode (str): Mode to run - "train" or "test".
        device (str): Device to use - "cpu" or "cuda:0", "cuda:1", "cuda:2".
        seed (int): Random seed for reproducibility.
        optimizer (str): Optimizer type.
        learning_rate (float): Learning rate for optimizer.
        momentum (float): Momentum for SGD optimizer.
        l2 (float): L2 regularization.
        lr_decay (float): Learning rate decay.
        batch_size (int): Batch size.
        num_epochs (int): Number of training epochs.
        max_no_incre (int): Early stopping criterion.
        max_grad_norm (float): Maximum gradient norm (for clipping).
        fp16 (int): Whether to use 16-bit floating point precision.
        model_folder (str): Directory to save model files.
        hidden_dim (int): Hidden size of the LSTM.
        dropout (float): Dropout rate.
        num_workers (int): Number of workers to load data
        embedder_type (str): Pretrained model type.
        add_iobes_constraint (int): Add IOBES constraint for transition parameters.
        print_detail_f1 (int): Print F1 scores for each tag.
        earlystop_atr (str): Metric for early stopping (micro or macro F1 score).
        test_file (str): Test file path for test mode.
    """

    if language not in ["en", "el", "ro", "bg"]:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are 'en', 'el', 'ro', and 'bg'."
        )

    if language == "en":
        main_checkpoint_path = f"{model_checkpoints_dir}/english_named_entity_recognition"
    elif language == "el":
        main_checkpoint_path = f"{model_checkpoints_dir}/greek_named_entity_recognition"
    elif language == "ro":
        main_checkpoint_path = f"{model_checkpoints_dir}/romanian_named_entity_recognition"
    elif language == "bg":
        main_checkpoint_path = f"{model_checkpoints_dir}/bulgarian_named_entity_recognition"
    else:
        raise ValueError(
            f"Unsupported language: {language}. Supported languages are 'en', 'el', 'ro', and 'bg'."
        )

    conf = Config(
        language=language,
        dataset=dataset,
        num_epochs=num_epochs,
        model_folder=main_checkpoint_path,
    )
    # Step 1: Set random seed for reproducibility
    set_seed(conf.seed)

    # Log the type of tokenizer being used
    logger.info(f"[Data Info] Tokenizing the instances using '{conf.embedder_type}' tokenizer")

    # Load a pre-trained tokenizer from HuggingFace
    tokenizer = AutoTokenizer.from_pretrained(
        conf.embedder_type,
        add_prefix_space=True,
        use_fast=True,
    )

    df, label2id, id2label = preprocess_dataset(dataset=dataset, language=language)
    conf.label2id = label2id
    conf.id2label = id2label

    # Define the sizes for train, validation, and test splits
    train_size = 0.8
    val_size = 0.1  # Validation size (10%)
    test_size = 0.1  # Test size (10%)

    # Split the dataset
    train_dataset = df.sample(frac=train_size)
    remaining_dataset = df.drop(train_dataset.index).reset_index(drop=True)

    # Now split the remaining dataset into validation and test
    val_dataset = remaining_dataset.sample(frac=val_size / (val_size + test_size), random_state=200)
    test_dataset = remaining_dataset.drop(val_dataset.index).reset_index(drop=True)

    # Reset index for all datasets
    train_dataset = train_dataset.reset_index(drop=True)
    val_dataset = val_dataset.reset_index(drop=True)
    test_dataset = test_dataset.reset_index(drop=True)

    # Print dataset shapes
    print("FULL Dataset: {}".format(df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("VALIDATION Dataset: {}".format(val_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Create dataset objects for training, validation, and testing
    training_set = NERDataset(train_dataset, tokenizer, 128, label2id=label2id)
    validation_set = NERDataset(val_dataset, tokenizer, 128, label2id=label2id)
    testing_set = NERDataset(test_dataset, tokenizer, 128, label2id=label2id)

    # # print the first 30 tokens and corresponding labels
    # for token, label in zip(tokenizer.convert_ids_to_tokens(training_set[0]["ids"][:30]),
    #                         training_set[0]["targets"][:30]):
    #     print('{0:10}  {1}'.format(token, id2label[label.item()]))

    # Create DataLoaders for training, validation, and test datasets
    train_dataloader, dev_dataloader, test_dataloader = create_dataloaders(
        training_set, validation_set, testing_set, conf
    )

    # Start training the model
    training_history, best_dev_f1, classification_report = train_model(
        conf,
        conf.num_epochs,
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        logger,
        id2label=id2label,
        label2id=label2id,
        main_checkpoint_path=main_checkpoint_path,
        tokenizer=tokenizer,
    )
    return training_history, best_dev_f1, classification_report


if __name__ == "__main__":
    training_history, best_dev_f1, classification_report = ner_training_service(
        # dataset="bg_ner_dataset.csv",
        # language="bg",
        # dataset="elNER4/elNER4.csv",
        # language="el",
        dataset="conll2003.csv",
        language="en",
        num_epochs=1,
    )
    # # assert training_history is not None
    # training_history, best_dev_f1, classification_report = ner_training_service(
    #     # dataset="bg_ner_dataset.csv",
    #     # language="bg",
    #     # dataset="elNER4/elNER4.csv",
    #     # language="el",
    #     dataset="conll2003.csv",
    #     # dataset="twitter_ner_data.csv",
    #     language="en",
    #     num_epochs=2,
    # )

    training_history, best_dev_f1, classification_report = ner_training_service(
        # dataset="bg_ner_dataset.csv",
        # language="bg",
        dataset="elNER4/elNER4.csv",
        language="el",
        # dataset="conll2003/conll2003.csv",
        # dataset="twitter_ner_data.csv",
        # language="en",
        num_epochs=2,
    )
    assert training_history is not None
    training_history, best_dev_f1, classification_report = ner_training_service(
        # dataset="bg_ner_dataset.csv",
        # language="bg",
        dataset="elNER18/elNER18.csv",
        language="el",
        # dataset="conll2003/conll2003.csv",
        # dataset="twitter_ner_data.csv",
        # language="en",
        num_epochs=2,
    )
    assert training_history is not None
    training_history, best_dev_f1, classification_report = ner_training_service(
        # dataset="bg_ner_dataset.csv",
        # language="bg",
        dataset="elNER4/elNER4.csv",
        language="el",
        # dataset="conll2003/conll2003.csv",
        # dataset="twitter_ner_data.csv",
        # language="en",
        num_epochs=2,
    )
    assert training_history is not None
