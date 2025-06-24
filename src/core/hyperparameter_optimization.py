import optuna

class HyperparameterOptimizer:
    def __init__(self, ensemble_factory_fn):
        """
        Parameters:
        - ensemble_factory_fn: a function that accepts a dictionary of hyperparameters
                               and returns a configured ensemble model instance.
        """
        self.study = None
        self.best_params = None
        self.ensemble_factory_fn = ensemble_factory_fn

    def optimize_ensemble_hyperparameters(self, train_dataset, val_dataset, n_trials=100):
        """Optimize ensemble model hyperparameters using Optuna."""

        def objective(trial):
            # RoBERTa hyperparameters
            roberta_lr = trial.suggest_float('roberta_lr', 1e-6, 5e-5, log=True)
            roberta_batch_size = trial.suggest_categorical('roberta_batch_size', [8, 16, 32])
            roberta_warmup_ratio = trial.suggest_float('roberta_warmup_ratio', 0.1, 0.3)
            roberta_weight_decay = trial.suggest_float('roberta_weight_decay', 0.01, 0.1)

            # GPT-4o hyperparameters
            gpt4o_temperature = trial.suggest_float('gpt4o_temperature', 0.1, 0.5)
            gpt4o_max_tokens = trial.suggest_int('gpt4o_max_tokens', 500, 1500)

            # Ensemble hyperparameters
            roberta_weight = trial.suggest_float('roberta_weight', 0.3, 0.7)
            confidence_threshold = trial.suggest_float('confidence_threshold', 0.6, 0.9)
            disagreement_threshold = trial.suggest_float('disagreement_threshold', 0.3, 0.7)

            # Create and configure ensemble
            ensemble = self.ensemble_factory_fn({
                'roberta_lr': roberta_lr,
                'roberta_batch_size': roberta_batch_size,
                'roberta_warmup_ratio': roberta_warmup_ratio,
                'roberta_weight_decay': roberta_weight_decay,
                'gpt4o_temperature': gpt4o_temperature,
                'gpt4o_max_tokens': gpt4o_max_tokens,
                'roberta_weight': roberta_weight,
                'confidence_threshold': confidence_threshold,
                'disagreement_threshold': disagreement_threshold
            })

            # Train and evaluate
            ensemble.fit(train_dataset)
            val_results = ensemble.evaluate(val_dataset)

            return val_results['f1_score']

        # Create and run Optuna study
        self.study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(),
            pruner=optuna.pruners.MedianPruner()
        )

        self.study.optimize(objective, n_trials=n_trials)

        # Save best parameters
        self.best_params = self.study.best_params

        return {
            'best_params': self.best_params,
            'best_score': self.study.best_value,
            'optimization_history': self.study.trials_dataframe()
        }


class EnsembleWeightOptimizer:
    def __init__(self):
        self.weight_history = []

    def optimize_weights(self, ensemble_system, optimization_dataset, n_trials=50):
        """Optimize ensemble weights using Optuna (replacing skopt)"""

        def objective(trial):
            # Define search space
            roberta_base_weight = trial.suggest_float('roberta_base_weight', 0.2, 0.8)
            gpt4o_weight = trial.suggest_float('gpt4o_weight', 0.2, 0.8)
            confidence_threshold = trial.suggest_float('confidence_threshold', 0.5, 0.95)
            disagreement_penalty = trial.suggest_float('disagreement_penalty', 0.1, 0.5)

            # Apply weights to the ensemble system
            ensemble_system.set_weights({
                'roberta_base_weight': roberta_base_weight,
                'gpt4o_weight': gpt4o_weight,
                'confidence_threshold': confidence_threshold,
                'disagreement_penalty': disagreement_penalty
            })

            # Evaluate ensemble
            results = ensemble_system.evaluate_batch(optimization_dataset)

            # Return F1 for maximization
            return results['f1_score']

        # Create Optuna study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        # Store and return best parameters
        best_params = study.best_params
        self.weight_history.append(best_params)

        return best_params
