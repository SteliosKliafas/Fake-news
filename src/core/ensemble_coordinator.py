class EnsembleCoordinator:
    def __init__(self):
        self.weight_optimizer = WeightOptimizer()
        self.confidence_calibrator = ConfidenceCalibrator()
        self.disagreement_resolver = DisagreementResolver()

    def combine_predictions(self, roberta_result: Dict, gpt4o_result: Dict,
                            text: str, domain: str) -> Dict:
        """Advanced ensemble combination with dynamic weighting"""

        # Extract predictions and confidences
        roberta_pred = roberta_result['prediction']
        roberta_conf = roberta_result['confidence']
        gpt4o_pred = gpt4o_result['prediction']
        gpt4o_conf = gpt4o_result['confidence']

        # Calculate dynamic weights based on multiple factors
        weights = self._calculate_dynamic_weights(
            roberta_result, gpt4o_result, text, domain
        )

        # Handle agreement vs disagreement
        if roberta_pred == gpt4o_pred:
            # Models agree - use confidence-weighted average
            final_confidence = (
                    weights['roberta'] * roberta_conf +
                    weights['gpt4o'] * gpt4o_conf
            )
            final_prediction = roberta_pred
            agreement_type = 'FULL_AGREEMENT'

        else:
            # Models disagree - use advanced resolution
            resolution = self.disagreement_resolver.resolve(
                roberta_result, gpt4o_result, text, domain, weights
            )
            final_prediction = resolution['prediction']
            final_confidence = resolution['confidence']
            agreement_type = 'DISAGREEMENT_RESOLVED'

        # Calibrate final confidence
        calibrated_confidence = self.confidence_calibrator.calibrate(
            final_confidence, final_prediction, domain
        )

        return {
            'prediction': final_prediction,
            'confidence': calibrated_confidence,
            'agreement_type': agreement_type,
            'model_weights': weights,
            'ensemble_method': 'DYNAMIC_WEIGHTED_VOTING',
            'meta_features': self._extract_meta_features(text, domain)
        }

    def _calculate_dynamic_weights(self, roberta_result: Dict, gpt4o_result: Dict,
                                   text: str, domain: str) -> Dict:
        """Calculate dynamic weights based on multiple factors"""

        # Base weights from historical performance
        base_weights = {
            'roberta': 0.6,  # Generally strong on linguistic patterns
            'gpt4o': 0.4  # Strong on reasoning and context
        }

        # Adjust based on text characteristics
        text_length = len(text.split())
        complexity_score = self._calculate_text_complexity(text)

        # RoBERTa performs better on shorter, pattern-based texts
        if text_length < 100:
            base_weights['roberta'] += 0.1
            base_weights['gpt4o'] -= 0.1

        # GPT-4o performs better on complex reasoning tasks
        if complexity_score > 0.7:
            base_weights['gpt4o'] += 0.15
            base_weights['roberta'] -= 0.15

        # Domain-specific adjustments
        domain_adjustments = {
            'political': {'roberta': 0.05, 'gpt4o': -0.05},  # RoBERTa better on political
            'health': {'roberta': -0.1, 'gpt4o': 0.1},  # GPT-4o better on health
            'technology': {'roberta': 0.0, 'gpt4o': 0.0}  # Equal performance
        }

        if domain in domain_adjustments:
            for model, adjustment in domain_adjustments[domain].items():
                base_weights[model] += adjustment

        # Confidence-based adjustments
        conf_diff = abs(roberta_result['confidence'] - gpt4o_result['confidence'])
        if conf_diff > 0.3:
            # Boost weight of more confident model
            if roberta_result['confidence'] > gpt4o_result['confidence']:
                base_weights['roberta'] += 0.1
                base_weights['gpt4o'] -= 0.1
            else:
                base_weights['gpt4o'] += 0.1
                base_weights['roberta'] -= 0.1

        # Normalize weights
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}

        return normalized_weights


class DisagreementResolver:
    def __init__(self):
        self.tie_breaker_model = self._load_tie_breaker_model()

    def resolve(self, roberta_result: Dict, gpt4o_result: Dict,
                text: str, domain: str, weights: Dict) -> Dict:
        """Resolve disagreements between models"""

        # Method 1: Confidence-based resolution
        conf_diff = abs(roberta_result['confidence'] - gpt4o_result['confidence'])

        if conf_diff > 0.4:  # Strong confidence difference
            if roberta_result['confidence'] > gpt4o_result['confidence']:
                return {
                    'prediction': roberta_result['prediction'],
                    'confidence': roberta_result['confidence'] * 0.9,  # Slightly reduce due to disagreement
                    'resolution_method': 'CONFIDENCE_BASED'
                }
            else:
                return {
                    'prediction': gpt4o_result['prediction'],
                    'confidence': gpt4o_result['confidence'] * 0.9,
                    'resolution_method': 'CONFIDENCE_BASED'
                }

        # Method 2: Weight-based resolution
        if weights['roberta'] > weights['gpt4o'] + 0.15:
            return {
                'prediction': roberta_result['prediction'],
                'confidence': roberta_result['confidence'] * 0.85,
                'resolution_method': 'WEIGHT_BASED'
            }
        elif weights['gpt4o'] > weights['roberta'] + 0.15:
            return {
                'prediction': gpt4o_result['prediction'],
                'confidence': gpt4o_result['confidence'] * 0.85,
                'resolution_method': 'WEIGHT_BASED'
            }

        # Method 3: Tie-breaker model
        tie_breaker_result = self._use_tie_breaker(text, roberta_result, gpt4o_result)

        return {
            'prediction': tie_breaker_result['prediction'],
            'confidence': tie_breaker_result['confidence'] * 0.8,  # Conservative confidence
            'resolution_method': 'TIE_BREAKER_MODEL'
        }