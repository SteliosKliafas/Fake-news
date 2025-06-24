import wandb
import mlflow
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


class ProductionMonitor:
    def __init__(self):
        # Metrics
        self.prediction_counter = Counter('predictions_total', 'Total predictions', ['model', 'domain', 'result'])
        self.prediction_latency = Histogram('prediction_duration_seconds', 'Prediction latency')
        self.confidence_gauge = Gauge('confidence_score', 'Average confidence score')
        self.accuracy_gauge = Gauge('rolling_accuracy', 'Rolling accuracy over time')

        # MLflow tracking
        mlflow.set_tracking_uri("http://mlflow-server:5000")

        # Wandb monitoring
        wandb.init(project="fake-news-production", job_type="monitoring")

        # Data storage
        self.prediction_log = []
        self.performance_metrics = {}

        # Start Prometheus metrics server
        start_http_server(8001)

    def log_prediction(self, response: PredictionResponse):
        """Log individual prediction for monitoring"""

        # Update Prometheus metrics
        self.prediction_counter.labels(
            model=response.model_used,
            domain=response.meta_data.get('domain', 'unknown'),
            result=response.prediction
        ).inc()

        self.prediction_latency.observe(response.processing_time_ms / 1000)
        self.confidence_gauge.set(response.confidence)

        # Log to internal storage
        log_entry = {
            'timestamp': datetime.now(),
            'prediction_id': response.prediction_id,
            'prediction': response.prediction,
            'confidence': response.confidence,
            'processing_time_ms': response.processing_time_ms,
            'model_used': response.model_used,
            'domain': response.meta_data.get('domain'),
            'text_length': len(response.meta_data.get('text', '')),
            'cache_hit': response.meta_data.get('cache_hit', False)
        }

        self.prediction_log.append(log_entry)

        # Log to Wandb
        wandb.log({
            'prediction_confidence': response.confidence,
            'processing_time_ms': response.processing_time_ms,
            'prediction_fake_rate': 1 if response.prediction == 'FAKE' else 0
        })

        # Periodic analysis
        if len(self.prediction_log) % 1000 == 0:
            self._analyze_performance_trends()

    def _analyze_performance_trends(self):
        """Analyze performance trends and detect issues"""

        if len(self.prediction_log) < 100:
            return

        recent_logs = self.prediction_log[-1000:]
        df = pd.DataFrame(recent_logs)

        # Calculate metrics
        avg_confidence = df['confidence'].mean()
        avg_latency = df['processing_time_ms'].mean()
        fake_rate = (df['prediction'] == 'FAKE').mean()
        cache_hit_rate = df['cache_hit'].mean()

        # Detect anomalies
        confidence_trend = self._detect_trend(df['confidence'].values)
        latency_trend = self._detect_trend(df['processing_time_ms'].values)

        # Alert conditions
        alerts = []

        if avg_confidence < 0.7:
            alerts.append("Low confidence detected")

        if avg_latency > 200:  # 200ms threshold
            alerts.append("High latency detected")

        if confidence_trend == 'decreasing':
            alerts.append("Confidence trend decreasing")

        if latency_trend == 'increasing':
            alerts.append("Latency trend increasing")

        # Log metrics
        metrics = {
            'avg_confidence': avg_confidence,
            'avg_latency_ms': avg_latency,
            'fake_rate': fake_rate,
            'cache_hit_rate': cache_hit_rate,
            'alerts': alerts
        }

        # Update Prometheus gauges
        self.confidence_gauge.set(avg_confidence)

        # Log to MLflow
        with mlflow.start_run():
            mlflow.log_metrics(metrics)

        # Log to Wandb
        wandb.log(metrics)

        # Send alerts if needed
        if alerts:
            self._send_alerts(alerts, metrics)

    def _detect_trend(self, values: np.ndarray, window: int = 50) -> str:
        """Detect trend in time series data"""

        if len(values) < window * 2:
            return 'insufficient_data'

        recent = values[-window:]
        previous = values[-2 * window:-window]

        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)

        change_percent = (recent_mean - previous_mean) / previous_mean * 100

        if change_percent > 5:
            return 'increasing'
        elif change_percent < -5:
            return 'decreasing'
        else:
            return 'stable'

    def _send_alerts(self, alerts: List[str], metrics: Dict):
        """Send alerts to monitoring systems"""

        alert_message = f"""
        🚨 Fake News Detection System Alert

        Alerts: {', '.join(alerts)}

        Current Metrics:
        - Average Confidence: {metrics['avg_confidence']:.3f}
        - Average Latency: {metrics['avg_latency_ms']:.1f}ms
        - Fake Rate: {metrics['fake_rate']:.2%}
        - Cache Hit Rate: {metrics['cache_hit_rate']:.2%}

        Time: {datetime.now()}
        """

        # Send to logging
        logging.warning(alert_message)

        # Send to Slack/Discord/Email (implement based on your needs)
        # self._send_slack_alert(alert_message)


class ModelDriftDetector:
    def __init__(self):
        self.reference_data = None
        self.drift_threshold = 0.1

    def set_reference_data(self, reference_predictions: List[Dict]):
        """Set reference data for drift detection"""
        self.reference_data = pd.DataFrame(reference_predictions)

    def detect_drift(self, current_predictions: List[Dict]) -> Dict:
        """Detect model drift using statistical tests"""

        if self.reference_data is None:
            return {'drift_detected': False, 'reason': 'No reference data'}

        current_data = pd.DataFrame(current_predictions)

        # Feature drift detection
        feature_drift = self._detect_feature_drift(current_data)

        # Prediction drift detection
        prediction_drift = self._detect_prediction_drift(current_data)

        # Performance drift detection
        performance_drift = self._detect_performance_drift(current_data)

        drift_detected = (
                feature_drift['drift'] or
                prediction_drift['drift'] or
                performance_drift['drift']
        )

        return {
            'drift_detected': drift_detected,
            'feature_drift': feature_drift,
            'prediction_drift': prediction_drift,
            'performance_drift': performance_drift,
            'timestamp': datetime.now()
        }

    def _detect_feature_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect drift in input features"""
        from scipy.stats import ks_2samp

        # Compare text length distributions
        ref_lengths = self.reference_data['text_length']
        cur_lengths = current_data['text_length']

        ks_stat, p_value = ks_2samp(ref_lengths, cur_lengths)

        return {
            'drift': p_value < 0.05,
            'ks_statistic': ks_stat,
            'p_value': p_value,
            'test': 'kolmogorov_smirnov'
        }

    def _detect_prediction_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect drift in prediction distributions"""

        ref_fake_rate = (self.reference_data['prediction'] == 'FAKE').mean()
        cur_fake_rate = (current_data['prediction'] == 'FAKE').mean()

        drift_magnitude = abs(ref_fake_rate - cur_fake_rate)

        return {
            'drift': drift_magnitude > self.drift_threshold,
            'reference_fake_rate': ref_fake_rate,
            'current_fake_rate': cur_fake_rate,
            'drift_magnitude': drift_magnitude
        }

    def _detect_performance_drift(self, current_data: pd.DataFrame) -> Dict:
        """Detect drift in model performance"""

        ref_confidence = self.reference_data['confidence'].mean()
        cur_confidence = current_data['confidence'].mean()

        confidence_drop = ref_confidence - cur_confidence

        return {
            'drift': confidence_drop > 0.1,  # 10% confidence drop threshold
            'reference_confidence': ref_confidence,
            'current_confidence': cur_confidence,
            'confidence_drop': confidence_drop
        }