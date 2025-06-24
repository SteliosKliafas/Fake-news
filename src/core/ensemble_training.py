from typing import Dict
from transformers import (
    RobertaTokenizerFast,  # safer and faster tokenizer, but if you prefer RobertaTokenizer, replace it here and below
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    RobertaForMaskedLM
)
from roberta_finetune import AdvancedRoBERTaTrainer
from gpt4o_finetune import GPT4oFinetuningPipeline
from hyperparameter_optimization import EnsembleWeightOptimizer, HyperparameterOptimizer
from ensemble_architecture import RoBERTaGPT4oEnsemble
from sklearn.calibration import CalibratedClassifierCV
from textattack import Attacker, AttackArgs
from textattack.datasets import Dataset
from textattack.models.wrappers import HuggingFaceModelWrapper
from textattack.attack_recipes import TextFoolerJin2019, BERTAttackLi2020, PWWSRen2019
from transformers import AutoTokenizer, AutoModelForSequenceClassification


from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset, concatenate_datasets

class AdversarialTrainer:
    def train_robust_model(self, model_path, adversarial_examples, clean_examples):
        """Fine-tunes a model on clean + adversarial data."""
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)

        def tokenize(batch):
            return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

        # Combine clean and adversarial examples
        combined_data = clean_examples + adversarial_examples
        dataset = Dataset.from_list(combined_data).shuffle(seed=42).map(tokenize, batched=True)

        # Basic train/validation split
        train_dataset = dataset.select(range(int(0.9 * len(dataset))))
        eval_dataset = dataset.select(range(int(0.9 * len(dataset)), len(dataset)))

        training_args = TrainingArguments(
            output_dir='./tmp_robust_model',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=50,
            save_total_limit=1,
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )

        trainer.train()

        return model



class AdversarialExampleGenerator:
    def __init__(self):
        self.attack_methods = {
            'textfooler': TextFoolerJin2019,
            'bert_attack': BERTAttackLi2020,
            'pwws': PWWSRen2019
        }

    def generate_examples(self, datasets, methods=['textfooler']):
        """Generate adversarial examples per domain using selected attack methods.

        Args:
            datasets: Dict[str, List[Dict[str, Any]]] - domain → list of samples with 'text' and 'label'
            methods: List[str] - attack method names
        Returns:
            Dict[str, List[Dict[str, Any]]] - domain → list of adversarial samples
        """
        adversarial_examples = {}

        for domain, examples in datasets.items():
            print(f"\n[Generating Adversarial Examples for Domain: {domain}]")

            model_path = f'./roberta_{domain}'  # must match your training output
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)

            model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

            textattack_dataset = Dataset([(ex['text'], ex['label']) for ex in examples])

            combined_examples = []
            for method in methods:
                print(f"  Using attack method: {method}")
                attack_cls = self.attack_methods[method]
                attack = attack_cls.build(model_wrapper)

                attack_args = AttackArgs(
                    num_examples=min(50, len(examples)),  # keep it reasonable for debugging
                    disable_stdout=True,
                    random_seed=42
                )

                attacker = Attacker(attack, textattack_dataset, attack_args)
                results = attacker.attack_dataset()

                for result in results:
                    if result.perturbed_result is not None:
                        combined_examples.append({
                            'text': result.perturbed_result.attacked_text.text,
                            'label': result.original_result.ground_truth_output
                        })

            adversarial_examples[domain] = combined_examples

        return adversarial_examples


class ConfidenceCalibrator:
    def fit(self, ensemble_system, dataset):
        # You must define how to get logits and true labels from the ensemble system
        logits, labels = ensemble_system.get_logits_and_labels(dataset)

        calibrator = CalibratedClassifierCV(cv='prefit')
        calibrator.fit(logits, labels)
        return calibrator


class MultiStageTrainingPipeline:
    def __init__(self):
        self.stages = {
            'stage1_roberta_pretraining': self._stage1_roberta_pretraining,
            'stage2_roberta_finetuning': self._stage2_roberta_finetuning,
            'stage3_gpt4o_finetuning': self._stage3_gpt4o_finetuning,
            'stage4_ensemble_optimization': self._stage4_ensemble_optimization,
            'stage5_adversarial_training': self._stage5_adversarial_training
        }

    def execute_full_pipeline(self, datasets: Dict):
        """Execute complete multi-stage training pipeline"""
        results = {}

        for stage_name, stage_func in self.stages.items():
            print(f"\n{'=' * 60}")
            print(f"EXECUTING {stage_name.upper()}")
            print(f"{'=' * 60}")

            stage_result = stage_func(datasets, results)
            results[stage_name] = stage_result

            # Save checkpoint after each stage
            self._save_stage_checkpoint(stage_name, stage_result)

            # Validate stage completion
            if not self._validate_stage_result(stage_name, stage_result):
                raise Exception(f"Stage {stage_name} failed validation")

        return results

    def _stage1_roberta_pretraining(self, datasets: Dict, previous_results: Dict):
        """Stage 1: Domain-adaptive pretraining of RoBERTa"""
        from transformers import DataCollatorForLanguageModeling

        model = RobertaForMaskedLM.from_pretrained('roberta-large')
        tokenizer = RobertaTokenizerFast.from_pretrained('roberta-large')  # Changed to Fast tokenizer

        # Prepare domain-specific corpus (should return a Dataset object)
        domain_corpus = self._prepare_domain_corpus(datasets['raw_news_corpus'])

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=0.15
        )

        training_args = TrainingArguments(
            output_dir='./roberta_domain_pretraining',
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            warmup_steps=2000,
            weight_decay=0.01,
            learning_rate=1e-5,
            fp16=True,
            logging_steps=500,
            save_steps=5000,
            dataloader_num_workers=8,
            evaluation_strategy="no",  # no evaluation in pretraining by default
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=domain_corpus,
        )

        trainer.train()

        model.save_pretrained('./roberta_domain_adapted')
        tokenizer.save_pretrained('./roberta_domain_adapted')

        last_log = None
        # Sometimes trainer.state.log_history can be empty if no eval or logging occurred, so safe get:
        if trainer.state.log_history:
            last_log = trainer.state.log_history[-1]
        training_loss = last_log.get('loss', None) if last_log else None

        return {
            'model_path': './roberta_domain_adapted',
            'training_loss': training_loss,
            'status': 'completed'
        }

    def _stage2_roberta_finetuning(self, datasets: Dict, previous_results: Dict):
        """Stage 2: Fine-tune RoBERTa for classification"""

        base_model_path = previous_results['stage1_roberta_pretraining']['model_path']

        roberta_trainer = AdvancedRoBERTaTrainer()

        domain_models = {}

        for domain in ['general', 'political', 'health', 'technology']:
            print(f"Training RoBERTa for domain: {domain}")

            domain_dataset = datasets[f'{domain}_dataset']

            # Use base_model_path if needed inside AdvancedRoBERTaTrainer (not shown in your snippet)
            # You might want to pass the base_model_path to the trainer or reload the model there.

            trainer = roberta_trainer.train_with_advanced_techniques(
                domain_dataset['train'],
                domain_dataset['validation']
            )

            eval_results = trainer.evaluate(domain_dataset['test'])

            model_path = f'./roberta_{domain}'
            roberta_trainer.save_model(model_path)

            domain_models[domain] = {
                'model_path': model_path,
                'eval_results': eval_results,
                'f1_score': eval_results.get('eval_f1', None)
            }

        best_general_f1 = max([m['f1_score'] for m in domain_models.values() if m['f1_score'] is not None])

        return {
            'domain_models': domain_models,
            'best_general_f1': best_general_f1,
            'status': 'completed'
        }

    def _stage3_gpt4o_finetuning(self, datasets: Dict, previous_results: Dict):
        """Stage 3: Fine-tune GPT-4o models"""

        gpt4o_pipeline = GPT4oFinetuningPipeline()

        fine_tuned_models = {}

        for domain in ['general', 'political', 'health', 'technology']:
            print(f"Fine-tuning GPT-4o for domain: {domain}")

            instruction_data = gpt4o_pipeline.create_instruction_dataset(
                datasets[f'{domain}_dataset']['train']
            )

            fine_tuning_job = gpt4o_pipeline.fine_tune_gpt4o_advanced(
                instruction_data, domain
            )

            # Wait for completion - fix: job id might be 'fine_tuning_job.id' or 'fine_tuning_job.job_id'
            job_id = getattr(fine_tuning_job, 'id', None) or getattr(fine_tuning_job, 'job_id', None)
            if job_id is None:
                raise RuntimeError(f"Fine-tuning job ID not found for domain {domain}")

            model_id = gpt4o_pipeline.wait_for_completion(job_id)

            eval_results = gpt4o_pipeline.evaluate_model(
                model_id, datasets[f'{domain}_dataset']['test']
            )

            fine_tuned_models[domain] = {
                'model_id': model_id,
                'job_id': job_id,
                'eval_results': eval_results,
                'accuracy': eval_results.get('accuracy', None)
            }

        best_accuracy = max([m['accuracy'] for m in fine_tuned_models.values() if m['accuracy'] is not None])

        return {
            'fine_tuned_models': fine_tuned_models,
            'best_accuracy': best_accuracy,
            'status': 'completed'
        }

    def _stage4_ensemble_optimization(self, datasets: Dict, previous_results: Dict):
        """Stage 4: Optimize ensemble weights and coordination"""

        roberta_models = previous_results['stage2_roberta_finetuning']['domain_models']
        gpt4o_models = previous_results['stage3_gpt4o_finetuning']['fine_tuned_models']

        ensemble_system = RoBERTaGPT4oEnsemble()
        ensemble_system.load_models(roberta_models, gpt4o_models)

        optimization_dataset = datasets['ensemble_optimization']

        weight_optimizer = EnsembleWeightOptimizer()
        optimal_weights = weight_optimizer.optimize_weights(
            ensemble_system, optimization_dataset
        )

        confidence_calibrator = ConfidenceCalibrator()
        calibration_params = confidence_calibrator.fit(
            ensemble_system, optimization_dataset
        )

        # datasets['test'] should exist and be test set for ensemble eval
        ensemble_eval_results = self._evaluate_ensemble(
            ensemble_system, datasets['test'], optimal_weights
        )

        return {
            'optimal_weights': optimal_weights,
            'calibration_params': calibration_params,
            'ensemble_eval_results': ensemble_eval_results,
            'ensemble_accuracy': ensemble_eval_results.get('accuracy', None),
            'status': 'completed'
        }

    def _stage5_adversarial_training(self, datasets: Dict, previous_results: Dict):
        """Stage 5: Adversarial training and robustness enhancement"""

        ensemble_system = self._load_optimized_ensemble(previous_results)

        adversarial_generator = AdversarialExampleGenerator()
        adversarial_examples = adversarial_generator.generate_examples(
            datasets['train'], methods=['textfooler', 'bert_attack', 'pwws']
        )

        for domain, model_info in previous_results['stage2_roberta_finetuning']['domain_models'].items():
            print(f"Adversarial training for RoBERTa {domain}")

            adversarial_trainer = AdversarialTrainer()
            robust_model = adversarial_trainer.train_robust_model(
                model_info['model_path'],
                adversarial_examples[domain],
                datasets[f'{domain}_dataset']['train']
            )

            robust_model_path = f'./roberta_{domain}_robust'
            robust_model.save_pretrained(robust_model_path)

        robustness_results = self._test_ensemble_robustness(
            ensemble_system, adversarial_examples
        )

        return {
            'adversarial_examples_generated': sum(len(v) for v in adversarial_examples.values()) if isinstance(adversarial_examples, dict) else len(adversarial_examples),
            'robustness_results': robustness_results,
            'robust_accuracy': robustness_results.get('accuracy', None),
            'status': 'completed'
        }

    # -- Add your private helper methods or make sure they're defined --
    def _save_stage_checkpoint(self, stage_name, stage_result):
        import json
        with open(f'checkpoint_{stage_name}.json', 'w') as f:
            json.dump(stage_result, f, indent=2)

    def _validate_stage_result(self, stage_name, stage_result):
        # Simple example: check if 'status' == 'completed'
        return stage_result.get('status') == 'completed'

    # Dummy placeholders: you must implement these
    def _prepare_domain_corpus(self, raw_corpus):
        # Should return a Dataset or DatasetDict suitable for Trainer
        pass

    def _evaluate_ensemble(self, ensemble_system, test_dataset, weights):
        # Return dict with at least 'accuracy'
        pass

    def _load_optimized_ensemble(self, previous_results):
        # Return ensemble_system loaded with optimized weights and models
        pass

    def _test_ensemble_robustness(self, ensemble_system, adversarial_examples):
        # Return dict with 'accuracy' metric for adversarial robustness
        pass
