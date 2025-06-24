import logging
import os
import string

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, set_seed

from model_checkpoints import model_checkpoints_dir
from src.ml.named_entity_recognition.config.config import Config
from src.ml.named_entity_recognition.processing.training import inference_single_sentence

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def split_text_for_ner(text, max_length=256, overlap=50):
    """
    Splits long text into smaller chunks suitable for Named Entity Recognition (NER) processing,
    ensuring that words and potential entities are not cut in the middle.

    Args:
        text (str): The input text to be split.
        max_length (int): Maximum length of each chunk in characters.
        overlap (int): Number of words to overlap between chunks.

    Returns:
        list: A list of text chunks ready for NER processing.
    """
    # If text is already short enough, return it as a single chunk
    if len(text) <= max_length:
        return [text]

    # Split the text into sentences
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed max_length, save current chunk and start a new one
        if len(current_chunk) + len(sentence) + 1 > max_length:
            if current_chunk:
                chunks.append(current_chunk.strip())

            # If the sentence itself is longer than max_length, we need to break it at word boundaries
            if len(sentence) > max_length:
                words = sentence.split()
                current_chunk = ""

                for word in words:
                    if len(current_chunk) + len(word) + 1 > max_length - overlap:
                        chunks.append(current_chunk.strip())
                        # Keep overlap words for the next chunk
                        overlap_words = current_chunk.split()[
                            -min(overlap, len(current_chunk.split())) :
                        ]
                        current_chunk = " ".join(overlap_words) + " "

                    current_chunk += word + " "
            else:
                current_chunk = sentence + " "
        else:
            current_chunk += sentence + " "

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def merge_ner_results(chunk_results):
    """
    Merges NER results from multiple text chunks, handling overlapping entities.

    Args:
        chunk_results (list): List of NER results from each text chunk.

    Returns:
        dict: Merged dictionary of entities.
    """
    merged_results = {}
    seen_entities = set()

    # Extract all unique entities from all chunks
    for chunk_result in chunk_results:
        for entity_type, entities in chunk_result.items():
            if entity_type not in merged_results:
                merged_results[entity_type] = []

            for entity in entities:
                # Handle different entity formats
                if isinstance(entity, str):
                    # Assuming simple string format
                    entity_text = entity
                    entity_obj = entity
                elif isinstance(entity, dict) and "text" in entity:
                    # Dictionary with 'text' key
                    entity_text = entity["text"]
                    entity_obj = entity
                elif isinstance(entity, tuple) and len(entity) >= 1:
                    # Tuple format (text, possibly other attributes)
                    entity_text = entity[0]
                    entity_obj = entity
                else:
                    # Unknown format, use string representation
                    entity_text = str(entity)
                    entity_obj = entity

                # Create a unique identifier for the entity to avoid duplicates
                entity_key = f"{entity_text}_{entity_type}"

                if entity_key not in seen_entities:
                    merged_results[entity_type].append(entity_obj)
                    seen_entities.add(entity_key)

    return merged_results


def process_long_text(text, model, device, id2label, tokenizer, max_length=256, overlap=50):
    """
    Process long text by splitting into chunks and merging results.

    Args:
        text (str): The input text to process.
        model: The NER model.
        device: The computing device (CPU/GPU).
        id2label: ID to label mapping.
        tokenizer: The tokenizer for processing text.
        max_length (int): Maximum chunk length.
        overlap (int): Word overlap between chunks.

    Returns:
        dict: Merged NER results.
    """
    import logging

    logger = logging.getLogger(__name__)

    # Split the text into appropriate chunks
    chunks = split_text_for_ner(text, max_length, overlap)
    logger.info(f"Split long text into {len(chunks)} chunks")

    # Process each chunk
    chunk_results = []
    for i, chunk in enumerate(chunks):
        try:
            word_to_label_df = inference_single_sentence(model, device, id2label, tokenizer, chunk)
            # word_to_label_df = word_to_label_df[word_to_label_df['tag'] != 'O']
            result = convert_entities_to_dict(word_to_label_df)
            chunk_results.append(result)
            logger.info(f"Processed chunk {i + 1}/{len(chunks)}")
        except Exception as e:
            logger.error(f"Error processing chunk {i + 1}: {str(e)}")
            # Continue with other chunks even if one fails
            continue

    # Merge the results
    try:
        merged_results = merge_ner_results(chunk_results)
        logger.info(f"Successfully merged results from {len(chunk_results)} chunks")
        return merged_results
    except Exception as e:
        logger.error(f"Error merging results: {str(e)}")
        # If merging fails, return results from the first chunk as fallback
        if chunk_results:
            return chunk_results[0]
        else:
            # If all chunks failed, return empty result
            return {"PERSON": [], "ORG": [], "LOC": [], "MISC": []}


def convert_entities_to_dict(entity_list):
    entities = {"PERSON": set(), "ORG": set(), "LOC": set(), "MISC": set()}
    current_entity = []
    current_label = None

    label_mapping = {"PERSON": "PERSON", "ORG": "ORG", "LOC": "LOC", "MISC": "MISC"}

    for word, label in entity_list:
        clean_word = word.strip(string.punctuation)
        if label.startswith("B-"):
            if current_entity and current_label:
                mapped_label = label_mapping.get(current_label, None)
                if mapped_label:
                    entities[mapped_label].add(" ".join(current_entity))
            current_label = label[2:]
            current_entity = [clean_word]
        elif label.startswith("I-") and current_label:
            current_entity.append(clean_word)
        else:
            if current_entity and current_label:
                mapped_label = label_mapping.get(current_label, None)
                if mapped_label:
                    entities[mapped_label].add(" ".join(current_entity))
            current_entity = []
            current_label = None

    # Add any remaining entity
    if current_entity and current_label:
        mapped_label = label_mapping.get(current_label, None)
        if mapped_label:
            entities[mapped_label].add(" ".join(current_entity))

    # Convert sets to lists for convenience
    return {key: sorted(value) for key, value in entities.items()}


def ner_inference_service(
    language, sentence, num_epochs=1, max_chunk_length=256, chunk_overlap=50, handle_long_text=True
):
    """
    Main function for executing Named Entity Recognition (NER) model training and evaluation.
    This function manages two modes: 'train' and 'test'. Based on the mode, it either trains the model
    on the specified dataset or loads and evaluates a pre-trained model.

    Args:
        mode (str): Mode to run - "train" or "test".
        device (str): Device to use - "cpu" or "cuda:0", "cuda:1", "cuda:2".
        seed (int): Random seed for reproducibility.
        dataset (str): Dataset to use.
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
    # Check if CUDA is available and set the device accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print the device
    print(f"Using device: {device}")

    # Check for supported languages
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

    # Initialize the configuration object with the passed arguments
    conf = Config(
        language=language,
        dataset=None,
        num_epochs=num_epochs,
        model_folder=main_checkpoint_path,
    )
    embedder_type_safe = conf.embedder_type.replace("/", "_")

    torch_model_file = os.path.join(
        main_checkpoint_path, f"full_model_{embedder_type_safe}.pt"
    )  # Change extension to .pt for PyTorch models

    if os.path.exists(torch_model_file):
        # Load the model from a PyTorch file
        model = torch.load(torch_model_file, map_location=device)

        print(f"Model loaded from {torch_model_file}")

    else:
        # Initialize a new model if no file exists
        model = AutoModelForTokenClassification.from_pretrained(
            conf.embedder_type,
            num_labels=len(conf.id2label),
            id2label=conf.id2label,
            label2id=conf.label2id,
        )
        print("No existing model found. Initialized a new model.")

    conf.label2id = model.config.label2id
    conf.id2label = model.config.id2label

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

    # Handle long text processing
    if handle_long_text and len(sentence) > max_chunk_length:
        return process_long_text(
            sentence, model, device, conf.id2label, tokenizer, max_chunk_length, chunk_overlap
        )
    else:
        # Process single text directly
        word_to_label_df = inference_single_sentence(
            model, device, conf.id2label, tokenizer, sentence
        )
        # word_to_label_df = word_to_label_df[word_to_label_df['tag'] != 'O']
        formatted_output = convert_entities_to_dict(word_to_label_df)
        return formatted_output


def evaluate_ner(predictions, ground_truth):
    """
    Evaluates Named Entity Recognition (NER) performance and identifies misclassified entities.
    Preserves original case when comparing entities for evaluation.

    :param predictions: Dictionary with entity types as keys and lists of entities as values from the NER model.
    :param ground_truth: List of tuples (entity, label) as the correct dataset.
    :return: Dictionary with Precision, Recall, F1-score.
    """
    # Convert predictions dictionary to list of (entity, label) tuples
    prediction_tuples = []
    for entity_type, entities in predictions.items():
        for entity in entities:
            prediction_tuples.append((entity, entity_type))

    # Create normalized sets for matching but keep original case for display
    # Use a case-insensitive comparison for matching, but preserve original entities for display
    pred_lookup = {ent.lower().strip(): (ent, label) for ent, label in prediction_tuples}
    truth_lookup = {ent.lower().strip(): (ent, label) for ent, label in ground_truth}

    # Get the set of normalized entity texts (lowercase)
    normalized_pred_keys = set(pred_lookup.keys())
    normalized_truth_keys = set(truth_lookup.keys())

    # Find matches based on normalized text
    matched_keys = normalized_pred_keys & normalized_truth_keys
    false_positive_keys = normalized_pred_keys - normalized_truth_keys
    false_negative_keys = normalized_truth_keys - normalized_pred_keys

    # Collect true positives, false positives, and false negatives with original case
    true_positives = []
    misclassified_entities = []

    for key in matched_keys:
        pred_entity, pred_label = pred_lookup[key]
        truth_entity, truth_label = truth_lookup[key]

        if pred_label == truth_label:
            true_positives.append((pred_entity, pred_label))
        else:
            misclassified_entities.append((truth_entity, pred_label, truth_label))

    false_positives = [(pred_lookup[key][0], pred_lookup[key][1]) for key in false_positive_keys]
    false_negatives = [(truth_lookup[key][0], truth_lookup[key][1]) for key in false_negative_keys]

    # Calculate metrics with error handling to avoid division by zero
    tp = len(true_positives)
    fp = len(false_positives) + len(misclassified_entities)
    fn = len(false_negatives) + len(misclassified_entities)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Print debugging details
    print("\n===== DEBUGGING DETAILS =====")
    print(f"True Positives: {len(true_positives)}")
    print(f"False Positives ({len(false_positives)} extra incorrect predictions):")
    for item in false_positives:
        print(f"  - {item}")

    print(f"False Negatives ({len(false_negatives)} missed correct entities):")
    for item in false_negatives:
        print(f"  - {item}")

    print(f"Misclassified Entities ({len(misclassified_entities)}):")
    for ent, pred_label, true_label in misclassified_entities:
        print(f"  - '{ent}' labeled as '{pred_label}' but should be '{true_label}'")

    # Return results
    return {
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1-score": round(f1_score, 4),
        "True Positives": tp,
        "False Positives": fp,
        "False Negatives": fn,
        "Misclassified Entities": misclassified_entities,
    }


def evaluate_sentences(sentences, ground_truth, language):
    """
    Evaluates multiple sentences against their ground truth data.

    :param sentences: List of sentences to evaluate
    :param ground_truth: List of ground truth entity annotations
    :param language: Language code for the NER model
    :return: List of NER results for each sentence
    """
    results = []

    for idx, (sentence, truth) in enumerate(zip(sentences, ground_truth)):
        print(f"\n\n===== Evaluating Sentence {idx + 1} ({language}) =====")

        # Get NER predictions
        result = ner_inference_service(sentence=sentence, language=language)

        # Evaluate predictions against ground truth
        evaluation = evaluate_ner(result, truth)
        results.append(result)

        print("\n===== FINAL NER METRICS =====")
        for key, value in evaluation.items():
            if key == "Misclassified Entities":
                print(f"{key}: {len(value)} - See details above")
            else:
                print(f"{key}: {value}")

    return results


if __name__ == "__main__":
    # English sentences for NER
    english_sentences = [
        "In New York City, Elon Musk, the CEO of Tesla, met with officials from NASA to discuss the upcoming SpaceX "
        "launch. Meanwhile, in Paris, renowned chef Gordon Ramsay was awarded the Michelin Star, recognizing his "
        "contribution to the culinary industry. At the same time, Kylian Mbappé signed a contract with Al Nassr "
        "FC, marking a historic moment in Saudi Arabian football. During a press conference at the United Nations "
        "Headquarters, Secretary-General António Guterres addressed the global climate crisis, citing reports from "
        "Greenpeace and the World Health Organization (WHO). Over in Tokyo, the annual Anime Expo attracted thousands "
        "of fans, eager to see exclusive previews from studios like Studio Ghibli and Toei Animation. In the tech "
        "industry, Sundar Pichai, the CEO of Google, announced a groundbreaking AI partnership with OpenAI, "
        "aiming to enhance ChatGPT's capabilities. Meanwhile, Jeff Bezos revealed Amazon's latest drone delivery "
        "system at an event in Silicon Valley. Over in London, officials from Oxford University and Cambridge "
        "University unveiled new research on quantum computing, funded by IBM and the European Research Council. "
        "Meanwhile, in Los Angeles, actress Scarlett Johansson promoted her latest movie at the Hollywood Walk of "
        "Fame, while pop star Taylor Swift prepared for her worldwide Eras Tour. In Rio de Janeiro, Neymar joined a "
        "charity event hosted by the International Red Cross, supporting underprivileged communities in Brazil. "
        "Across the globe, political leaders like Joe Biden, Vladimir Putin, and Emmanuel Macron gathered in Brussels "
        "for a G7 Summit, discussing global economic policies and security measures. At the same time, in Beijing, "
        "representatives from Huawei and Alibaba met with government officials to strategize on future technological "
        "advancements. In the world of literature, J.K. Rowling released a new book under the Bloomsbury Publishing "
        "label, while in Berlin, the Frankfurt Book Fair showcased works from authors worldwide, including "
        "bestsellers by Stephen King and Haruki Murakami. Back in San Francisco, developers at Meta and Apple "
        "introduced new virtual reality technologies, with Mark Zuckerberg presenting updates on the Metaverse "
        "project. Meanwhile, in Mumbai, Bollywood icon Shah Rukh Khan attended the premiere of his latest film, "
        "co-produced by Yash Raj Films.",
        "At the World Economic Forum in Davos, Bill Gates, co-founder of Microsoft, discussed climate change "
        "solutions with world leaders including Angela Merkel, the former Chancellor of Germany. Meanwhile, "
        "in Hollywood, director Christopher Nolan premiered his latest film, which stars Leonardo DiCaprio and is "
        "distributed by Warner Bros. In London, the Bank of England raised interest rates following a meeting with "
        "the European Central Bank, led by Christine Lagarde. On the streets of Tokyo, fashion icon Rei Kawakubo "
        "showcased a groundbreaking collection at the Comme des Garçons runway, attended by notable figures such as "
        "Anna Wintour and Karl Lagerfeld. In the world of sports, Lionel Messi extended his contract with Paris "
        "Saint-Germain, while Serena Williams announced her retirement from professional tennis during an exclusive "
        "interview on ESPN. Over in Dubai, Sheikh Mohammed bin Rashid Al Maktoum unveiled plans for a futuristic "
        "smart city that will use AI to monitor urban systems. At the same time, in New Delhi, Prime Minister "
        "Narendra Modi attended the groundbreaking ceremony for the country's new spaceport, with officials from ISRO "
        "and SpaceX in attendance. In the world of technology, Tim Cook, CEO of Apple, revealed the new iPhone at an "
        "event held at Apple's headquarters in Cupertino. Meanwhile, in Moscow, President Vladimir Putin met with Xi "
        "Jinping to strengthen ties between Russia and China, signing several trade agreements. In Cairo, "
        "archaeologists uncovered a new tomb near the Great Pyramids of Giza, attracting scholars from Harvard "
        "University and the British Museum. In Paris, the Louvre hosted an exclusive exhibition of works by Pablo "
        "Picasso, curated by renowned art historian Robert Hughes. Lastly, in Rio de Janeiro, samba legend Gilberto "
        "Gil celebrated his 80th birthday with a concert at the Maracanã Stadium, attended by former Brazilian "
        "President Luiz Inácio Lula da Silva.",
    ]

    ground_truth = [
        [
            ("Elon Musk", "PERSON"),
            ("Tesla", "ORG"),
            ("NASA", "ORG"),
            ("SpaceX", "ORG"),
            ("New York City", "LOC"),
            ("Paris", "LOC"),
            ("Gordon Ramsay", "PERSON"),
            ("Michelin Star", "MISC"),
            ("Kylian Mbappé", "PERSON"),
            ("Al Nassr FC", "ORG"),
            ("Saudi Arabian football", "MISC"),
            ("United Nations Headquarters", "LOC"),
            ("António Guterres", "PERSON"),
            ("Greenpeace", "ORG"),
            ("World Health Organization (WHO)", "ORG"),
            ("Tokyo", "LOC"),
            ("Anime Expo", "MISC"),
            ("Studio Ghibli", "ORG"),
            ("Toei Animation", "ORG"),
            ("Sundar Pichai", "PERSON"),
            ("Google", "ORG"),
            ("OpenAI", "ORG"),
            ("ChatGPT", "MISC"),
            ("Jeff Bezos", "PERSON"),
            ("Amazon", "ORG"),
            ("Silicon Valley", "LOC"),
            ("Oxford University", "ORG"),
            ("Cambridge University", "ORG"),
            ("IBM", "ORG"),
            ("World Health Organization", "ORG"),
            ("European Research Council", "ORG"),
            ("London", "LOC"),
            ("Scarlett Johansson", "PERSON"),
            ("Hollywood Walk of Fame", "MISC"),
            ("Taylor Swift", "PERSON"),
            ("Eras Tour", "MISC"),
            ("Rio de Janeiro", "LOC"),
            ("Neymar", "PERSON"),
            ("International Red Cross", "ORG"),
            ("Joe Biden", "PERSON"),
            ("Vladimir Putin", "PERSON"),
            ("Emmanuel Macron", "PERSON"),
            ("Brussels", "LOC"),
            ("G7 Summit", "MISC"),
            ("Beijing", "LOC"),
            ("Huawei", "ORG"),
            ("Alibaba", "ORG"),
            ("J.K. Rowling", "PERSON"),
            ("Bloomsbury Publishing", "ORG"),
            ("Berlin", "LOC"),
            ("Frankfurt Book Fair", "MISC"),
            ("Stephen King", "PERSON"),
            ("Haruki Murakami", "PERSON"),
            ("San Francisco", "LOC"),
            ("Meta", "ORG"),
            ("Apple", "ORG"),
            ("Mark Zuckerberg", "PERSON"),
            ("Metaverse", "MISC"),
            ("Mumbai", "LOC"),
            ("Shah Rukh Khan", "PERSON"),
            ("Brazil", "LOC"),
            ("Yash Raj Films", "ORG"),
            ("Los Angeles", "LOC"),
            ("Bollywood", "MISC"),
            ("United Nations", "ORG"),
            ("WHO", "ORG"),
        ],
        [
            ("Bill Gates", "PERSON"),
            ("Microsoft", "ORG"),
            ("World Economic Forum", "MISC"),
            ("Sheikh Mohammed bin Rashid Al Maktoum", "PERSON"),
            ("Angela Merkel", "PERSON"),
            ("Germany", "LOC"),
            ("Leonardo DiCaprio", "PERSON"),
            ("Warner Bros", "ORG"),
            ("Hollywood", "MISC"),
            ("Dubai", "LOC"),
            ("Bank of England", "ORG"),
            ("European Central Bank", "ORG"),
            ("Christine Lagarde", "PERSON"),
            ("London", "LOC"),
            ("Rei Kawakubo", "PERSON"),
            ("Comme des Garçons", "ORG"),
            ("Tokyo", "LOC"),
            ("Anna Wintour", "PERSON"),
            ("Karl Lagerfeld", "PERSON"),
            ("Serena Williams", "PERSON"),
            ("ESPN", "ORG"),
            ("Paris Saint-Germain", "ORG"),
            ("Lionel Messi", "PERSON"),
            ("Cupertino", "LOC"),
            ("Tim Cook", "PERSON"),
            ("Apple", "ORG"),
            ("iPhone", "MISC"),
            ("Moscow", "LOC"),
            ("Vladimir Putin", "PERSON"),
            ("Xi Jinping", "PERSON"),
            ("Russia", "LOC"),
            ("China", "LOC"),
            ("New Delhi", "LOC"),
            ("Narendra Modi", "PERSON"),
            ("ISRO", "ORG"),
            ("SpaceX", "ORG"),
            ("Cairo", "LOC"),
            ("Harvard University", "ORG"),
            ("British Museum", "ORG"),
            ("Pablo Picasso", "PERSON"),
            ("Louvre", "ORG"),
            ("Robert Hughes", "PERSON"),
            ("Rio de Janeiro", "LOC"),
            ("Gilberto Gil", "PERSON"),
            ("Maracanã Stadium", "LOC"),
            ("Luiz Inácio Lula da Silva", "PERSON"),
            ("Great Pyramids of Giza", "LOC"),
            ("Christopher Nolan", "PERSON"),
            ("Paris", "LOC"),
            ("Warner Bros", "ORG"),
            ("Davos", "LOC"),
        ],
    ]

    greek_sentences = [
        """
        Στη Νέα Υόρκη, ο Έλον Μασκ, διευθύνων σύμβουλος της Τέσλα, συναντήθηκε με αξιωματούχους της NASA
        για να συζητήσουν την επερχόμενη εκτόξευση της SpaceX. Εν τω μεταξύ, στο Παρίσι, ο διάσημος σεφ
        Γκόρντον Ράμσεϊ βραβεύτηκε με αστέρι Μισελέν, αναγνωρίζοντας τη συμβολή του στη γαστρονομική
        βιομηχανία. Την ίδια στιγμή, ο Κιλιάν Εμπαπέ υπέγραψε συμβόλαιο με την Αλ Νασρ FC,
        σηματοδοτώντας μια ιστορική στιγμή για το ποδόσφαιρο της Σαουδικής Αραβίας. Κατά τη διάρκεια
        συνέντευξης Τύπου στα Κεντρικά των Ηνωμένων Εθνών, ο Γενικός Γραμματέας Αντόνιο Γκουτέρες
        αναφέρθηκε στην παγκόσμια κλιματική κρίση, επικαλούμενος εκθέσεις της Greenpeace και του
        Παγκόσμιου Οργανισμού Υγείας (ΠΟΥ). Στο Τόκιο, η ετήσια Έκθεση Anime προσέλκυσε χιλιάδες
        θαυμαστές, πρόθυμους να δουν αποκλειστικές προεπισκοπήσεις από στούντιο όπως το Studio Ghibli
        και το Toei Animation. Στον τομέα της τεχνολογίας, ο Σούνταρ Πιτσάι, διευθύνων σύμβουλος της
        Google, ανακοίνωσε μια πρωτοποριακή συνεργασία τεχνητής νοημοσύνης με την OpenAI, με στόχο την
        ενίσχυση των δυνατοτήτων του ChatGPT. Εν τω μεταξύ, ο Τζεφ Μπέζος αποκάλυψε το νεότερο σύστημα
        παράδοσης με drone της Amazon σε μια εκδήλωση στη Σίλικον Βάλεϊ. Στο Λονδίνο, αξιωματούχοι από
        το Πανεπιστήμιο της Οξφόρδης και το Πανεπιστήμιο του Κέιμπριτζ παρουσίασαν νέα έρευνα για την
        κβαντική υπολογιστική, χρηματοδοτούμενη από την IBM και το Ευρωπαϊκό Συμβούλιο Έρευνας. Εν τω
        μεταξύ, στο Λος Άντζελες, η ηθοποιός Σκάρλετ Γιόχανσον προώθησε την τελευταία της ταινία στο
        Χόλιγουντ Γουόκ οφ Φέιμ, ενώ η ποπ σταρ Τέιλορ Σουίφτ προετοιμαζόταν για την παγκόσμια περιοδεία
        της Eras Tour. Στο Ρίο ντε Τζανέιρο, ο Νεϊμάρ συμμετείχε σε φιλανθρωπική εκδήλωση που διοργάνωσε
        ο Διεθνής Ερυθρός Σταυρός, υποστηρίζοντας υποβαθμισμένες κοινότητες στη Βραζιλία.
        """,
        """
        Στο Παγκόσμιο Οικονομικό Φόρουμ στο Νταβός, ο Μπιλ Γκέιτς, συνιδρυτής της Microsoft, συζήτησε
        λύσεις για την κλιματική αλλαγή με παγκόσμιους ηγέτες, συμπεριλαμβανομένης της Άνγκελα Μέρκελ,
        πρώην Καγκελαρίου της Γερμανίας. Εν τω μεταξύ, στο Χόλιγουντ, ο σκηνοθέτης Κρίστοφερ Νόλαν
        παρουσίασε την πρεμιέρα της τελευταίας του ταινίας, με πρωταγωνιστή τον Λεονάρντο Ντι Κάπριο και
        διανομή από τη Warner Bros. Στο Λονδίνο, η Τράπεζα της Αγγλίας αύξησε τα επιτόκια μετά από
        συνάντηση με την Ευρωπαϊκή Κεντρική Τράπεζα, της οποίας ηγείται η Κριστίν Λαγκάρντ. Στους
        δρόμους του Τόκιο, η σχεδιάστρια μόδας Ρέι Καβακούμπο παρουσίασε μια πρωτοποριακή συλλογή στην
        πασαρέλα της Comme des Garçons, που παρακολούθησαν σημαντικές προσωπικότητες όπως η Άννα
        Γουίντουρ και ο Καρλ Λάγκερφελντ. Στον κόσμο του αθλητισμού, ο Λιονέλ Μέσι επέκτεινε το
        συμβόλαιό του με την Παρί Σεν Ζερμέν, ενώ η Σερένα Γουίλιαμς ανακοίνωσε την αποχώρησή της από το
        επαγγελματικό τένις κατά τη διάρκεια αποκλειστικής συνέντευξης στο ESPN.
        """,
    ]

    greek_ground_truth = [
        # First Greek text entities
        [
            ("Νέα Υόρκη", "LOC"),
            ("Έλον Μασκ", "PERSON"),
            ("Τέσλα", "ORG"),
            ("NASA", "ORG"),
            ("SpaceX", "ORG"),
            ("Παρίσι", "LOC"),
            ("Γκόρντον Ράμσεϊ", "PERSON"),
            ("Μισελέν", "MISC"),
            ("Κιλιάν Εμπαπέ", "PERSON"),
            ("Αλ Νασρ FC", "ORG"),
            ("Σαουδικής Αραβίας", "LOC"),
            ("Ηνωμένων Εθνών", "ORG"),
            ("Αντόνιο Γκουτέρες", "PERSON"),
            ("Greenpeace", "ORG"),
            ("Παγκόσμιου Οργανισμού Υγείας", "ORG"),
            ("ΠΟΥ", "ORG"),
            ("Τόκιο", "LOC"),
            ("Έκθεση Anime", "MISC"),
            ("Studio Ghibli", "ORG"),
            ("Toei Animation", "ORG"),
            ("Σούνταρ Πιτσάι", "PERSON"),
            ("Google", "ORG"),
            ("OpenAI", "ORG"),
            ("ChatGPT", "MISC"),
            ("Τζεφ Μπέζος", "PERSON"),
            ("Amazon", "ORG"),
            ("Σίλικον Βάλεϊ", "LOC"),
            ("Λονδίνο", "LOC"),
            ("Πανεπιστήμιο της Οξφόρδης", "ORG"),
            ("Πανεπιστήμιο του Κέιμπριτζ", "ORG"),
            ("IBM", "ORG"),
            ("Ευρωπαϊκό Συμβούλιο Έρευνας", "ORG"),
            ("Λος Άντζελες", "LOC"),
            ("Σκάρλετ Γιόχανσον", "PERSON"),
            ("Χόλιγουντ Γουόκ οφ Φέιμ", "MISC"),
            ("Τέιλορ Σουίφτ", "PERSON"),
            ("Eras Tour", "MISC"),
            ("Ρίο ντε Τζανέιρο", "LOC"),
            ("Νεϊμάρ", "PERSON"),
            ("Διεθνής Ερυθρός Σταυρός", "ORG"),
            ("Βραζιλία", "LOC"),
        ],
        # Second Greek text entities
        [
            ("Παγκόσμιο Οικονομικό Φόρουμ", "MISC"),
            ("Νταβός", "LOC"),
            ("Μπιλ Γκέιτς", "PERSON"),
            ("Microsoft", "ORG"),
            ("Άνγκελα Μέρκελ", "PERSON"),
            ("Γερμανίας", "LOC"),
            ("Χόλιγουντ", "LOC"),
            ("Κρίστοφερ Νόλαν", "PERSON"),
            ("Λεονάρντο Ντι Κάπριο", "PERSON"),
            ("Warner Bros", "ORG"),
            ("Λονδίνο", "LOC"),
            ("Τράπεζα της Αγγλίας", "ORG"),
            ("Ευρωπαϊκή Κεντρική Τράπεζα", "ORG"),
            ("Κριστίν Λαγκάρντ", "PERSON"),
            ("Τόκιο", "LOC"),
            ("Ρέι Καβακούμπο", "PERSON"),
            ("Comme des Garçons", "ORG"),
            ("Άννα Γουίντουρ", "PERSON"),
            ("Καρλ Λάγκερφελντ", "PERSON"),
            ("Λιονέλ Μέσι", "PERSON"),
            ("Παρί Σεν Ζερμέν", "ORG"),
            ("Σερένα Γουίλιαμς", "PERSON"),
            ("ESPN", "ORG"),
        ],
    ]

    # Evaluate English sentences
    print("\n\n========== ENGLISH EVALUATION ==========")
    english_results = evaluate_sentences(english_sentences, ground_truth, language="en")

    # Evaluate Greek sentences
    print("\n\n========== GREEK EVALUATION ==========")
    greek_results = evaluate_sentences(greek_sentences, greek_ground_truth, language="el")

    # Combine results
    r = english_results + greek_results
    assert r is not None
