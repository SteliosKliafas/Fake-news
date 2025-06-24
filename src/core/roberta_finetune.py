import torch
import torch.nn as nn  # <-- Needed for nn.Module, nn.Linear, etc.
from transformers import RobertaTokenizer
from transformers import (
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from transformers.optimization import get_cosine_schedule_with_warmup

import torch.optim as optim  # if you use optimizers directly


from transformers import Trainer, TrainerCallback
import os

class AdvancedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    # You can override methods here if you want custom training behavior
    # For now, it uses the default Trainer behavior

class AdvancedLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print(f"[Step {state.global_step}] Logs: {logs}")
        # Extend this to log to wandb or other logging systems if needed

class ModelCheckpointCallback(TrainerCallback):
    def __init__(self, save_dir='./checkpoints', save_best_only=True, metric_name='eval_f1'):
        self.save_dir = save_dir
        self.save_best_only = save_best_only
        self.metric_name = metric_name
        self.best_metric = None
        os.makedirs(save_dir, exist_ok=True)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return

        current_metric = metrics.get(self.metric_name)
        if current_metric is None:
            return

        if (self.best_metric is None) or (current_metric > self.best_metric):
            self.best_metric = current_metric
            checkpoint_path = os.path.join(self.save_dir, f'checkpoint-{state.global_step}')
            print(f"New best {self.metric_name}: {current_metric:.4f}. Saving model to {checkpoint_path}")
            control.should_save = True
            control.output_dir = checkpoint_path
        else:
            control.should_save = False




class AdvancedRoBERTaTrainer:
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model_configs = {
            'base': {
                'model_name': 'roberta-base',
                'max_length': 512,
                'batch_size': 32,
                'learning_rate': 2e-5
            },
            'large': {
                'model_name': 'roberta-large',
                'max_length': 512,
                'batch_size': 16,
                'learning_rate': 1e-5
            }
        }

    def create_advanced_model(self, base_model: str = 'roberta-large'):
        """Create RoBERTa with advanced architectural modifications"""

        class AdvancedRoBERTaModel(nn.Module):
            def __init__(self, base_model_name):
                super().__init__()
                self.roberta = RobertaForSequenceClassification.from_pretrained(
                    base_model_name,
                    num_labels=2,
                    output_attentions=True,
                    output_hidden_states=True
                )

                # Additional layers for enhanced performance
                self.feature_extractor = nn.Sequential(
                    nn.Linear(1024, 512),  # roberta-large hidden size
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.ReLU(),
                    nn.Dropout(0.2)
                )

                # Multi-head classification
                self.classifiers = nn.ModuleDict({
                    'main': nn.Linear(256, 2),
                    'confidence': nn.Linear(256, 1),
                    'domain': nn.Linear(256, 5)  # politics, health, sports, etc.
                })

                # Attention mechanism for explanation
                self.explanation_attention = nn.MultiheadAttention(
                    embed_dim=256, num_heads=8, dropout=0.1
                )

            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.roberta(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    output_hidden_states=True
                )

                # Extract features
                pooled_output = outputs.pooler_output
                features = self.feature_extractor(pooled_output)

                # Multi-task predictions
                main_logits = self.classifiers['main'](features)
                confidence_score = torch.sigmoid(self.classifiers['confidence'](features))
                domain_logits = self.classifiers['domain'](features)

                # Calculate losses if labels provided
                loss = None
                if labels is not None:
                    loss_fct = nn.CrossEntropyLoss()
                    main_loss = loss_fct(main_logits, labels)

                    # Confidence loss (encourage high confidence for correct predictions)
                    predicted_probs = torch.softmax(main_logits, dim=-1)
                    max_probs = torch.max(predicted_probs, dim=-1)[0]
                    confidence_loss = nn.MSELoss()(confidence_score.squeeze(), max_probs)

                    loss = main_loss + 0.1 * confidence_loss

                return {
                    'loss': loss,
                    'logits': main_logits,
                    'confidence': confidence_score,
                    'domain_logits': domain_logits,
                    'attentions': outputs.attentions,
                    'hidden_states': outputs.hidden_states,
                    'features': features
                }

        return AdvancedRoBERTaModel(base_model)

    def train_with_advanced_techniques(self, train_dataset, val_dataset):
        """Training with cutting-edge techniques"""

        from transformers import TrainingArguments, Trainer
        from transformers.optimization import get_cosine_schedule_with_warmup
        import torch.optim as optim

        model = self.create_advanced_model()

        # Advanced training arguments
        training_args = TrainingArguments(
            output_dir='./advanced_roberta_results',
            num_train_epochs=5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=32,
            gradient_accumulation_steps=2,
            warmup_steps=1000,
            weight_decay=0.01,
            learning_rate=1e-5,

            # Advanced optimization
            fp16=True,
            gradient_checkpointing=True,
            dataloader_num_workers=8,

            # Evaluation and saving
            evaluation_strategy='steps',
            eval_steps=500,
            save_steps=1000,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,

            # Logging
            logging_steps=100,
            report_to=['wandb'],
            run_name='advanced_roberta_fake_news'
        )

        # Custom optimizer with layer-wise learning rates
        optimizer = self._create_layerwise_optimizer(model)

        # Advanced scheduler
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=1000,
            num_training_steps=len(
                train_dataset) * training_args.num_train_epochs // training_args.per_device_train_batch_size
        )

        # Custom trainer with advanced features
        trainer = AdvancedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self._compute_advanced_metrics,
            optimizers=(optimizer, scheduler),
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=3),
                AdvancedLoggingCallback(),
                ModelCheckpointCallback()
            ]
        )

        # Train with automatic mixed precision and gradient clipping
        trainer.train()

        return trainer

    def _create_layerwise_optimizer(self, model):
        """Create optimizer with layer-wise learning rates"""

        parameters = []

        # Lower learning rate for pre-trained layers
        for name, param in model.roberta.named_parameters():
            if 'embeddings' in name:
                parameters.append({'params': param, 'lr': 5e-6})
            elif 'encoder.layer' in name:
                layer_num = int(name.split('.')[3])
                # Gradually increase learning rate for higher layers
                lr = 5e-6 + (1e-5 - 5e-6) * (layer_num / 24)
                parameters.append({'params': param, 'lr': lr})
            else:
                parameters.append({'params': param, 'lr': 1e-5})

        # Higher learning rate for new layers
        for name, param in model.feature_extractor.named_parameters():
            parameters.append({'params': param, 'lr': 3e-5})

        for classifier in model.classifiers.values():
            for param in classifier.parameters():
                parameters.append({'params': param, 'lr': 3e-5})

        return torch.optim.AdamW(parameters, weight_decay=0.01)