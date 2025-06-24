from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import asyncio
import redis
import json
from typing import Optional, List
import time
import uuid

app = FastAPI(title="Advanced Fake News Detection API", version="2.0.0")


class PredictionRequest(BaseModel):
    text: str
    domain: Optional[str] = "general"
    use_ensemble: Optional[bool] = True
    require_explanation: Optional[bool] = True
    confidence_threshold: Optional[float] = 0.8


class PredictionResponse(BaseModel):
    prediction_id: str
    prediction: str
    confidence: float
    processing_time_ms: float
    model_used: str
    explanation: Optional[dict] = None
    uncertainty_metrics: Optional[dict] = None
    meta_data: dict


class AdvancedFakeNewsAPI:
    def __init__(self):
        # Load ensemble system
        self.ensemble_system = RoBERTaGPT4oEnsemble()
        self.ensemble_system.load_production_models()

        # Redis for caching and rate limiting
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)

        # Performance monitoring
        self.performance_monitor = ProductionMonitor()

        # Request queue for batch processing
        self.request_queue = asyncio.Queue(maxsize=1000)

        # Start background workers
        asyncio.create_task(self._batch_processing_worker())

    async def predict_single(self, request: PredictionRequest) -> PredictionResponse:
        """Single prediction with full ensemble"""

        start_time = time.time()
        prediction_id = str(uuid.uuid4())

        try:
            # Check cache first
            cache_key = f"prediction:{hash(request.text)}"
            cached_result = self.redis_client.get(cache_key)

            if cached_result:
                result = json.loads(cached_result)
                result['cache_hit'] = True
                return PredictionResponse(**result, prediction_id=prediction_id)

            # Make prediction
            result = await self.ensemble_system.predict(
                text=request.text,
                domain=request.domain,
                use_ensemble=request.use_ensemble
            )

            # Add explanation if requested
            if request.require_explanation:
                result['explanation'] = await self._generate_explanation(
                    request.text, result
                )

            processing_time = (time.time() - start_time) * 1000

            response = PredictionResponse(
                prediction_id=prediction_id,
                prediction=result['prediction'],
                confidence=result['confidence'],
                processing_time_ms=processing_time,
                model_used=result.get('model_used', 'ensemble'),
                explanation=result.get('explanation'),
                uncertainty_metrics=result.get('uncertainty_metrics'),
                meta_data={
                    'domain': request.domain,
                    'ensemble_used': request.use_ensemble,
                    'cache_hit': False
                }
            )

            # Cache result
            self.redis_client.setex(
                cache_key,
                3600,  # 1 hour TTL
                json.dumps(response.dict())
            )

            # Log for monitoring
            self.performance_monitor.log_prediction(response)

            return response

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def predict_batch(self, requests: List[PredictionRequest]) -> List[PredictionResponse]:
        """Batch prediction for high throughput"""

        # Add requests to queue
        batch_id = str(uuid.uuid4())

        for request in requests:
            await self.request_queue.put({
                'batch_id': batch_id,
                'request': request,
                'timestamp': time.time()
            })

        # Wait for batch completion
        batch_results = await self._wait_for_batch_completion(batch_id, len(requests))

        return batch_results

    async def _batch_processing_worker(self):
        """Background worker for batch processing"""

        while True:
            batch_items = []
            batch_size = 32

            # Collect batch items
            for _ in range(batch_size):
                try:
                    item = await asyncio.wait_for(self.request_queue.get(), timeout=1.0)
                    batch_items.append(item)
                except asyncio.TimeoutError:
                    break

            if batch_items:
                await self._process_batch(batch_items)

    async def _process_batch(self, batch_items):
        """Process a batch of requests efficiently"""

        # Group by domain for optimal processing
        domain_groups = {}
        for item in batch_items:
            domain = item['request'].domain
            if domain not in domain_groups:
                domain_groups[domain] = []
            domain_groups[domain].append(item)

        # Process each domain group
        results = {}

        for domain, items in domain_groups.items():
            texts = [item['request'].text for item in items]

            # Batch prediction
            domain_results = await self.ensemble_system.predict_batch(texts, domain)

            for i, item in enumerate(items):
                results[item['batch_id']] = results.get(item['batch_id'], [])
                results[item['batch_id']].append(domain_results[i])

        # Store results in Redis
        for batch_id, batch_results in results.items():
            self.redis_client.setex(
                f"batch_results:{batch_id}",
                300,  # 5 minutes TTL
                json.dumps(batch_results)
            )


# FastAPI endpoints
api = AdvancedFakeNewsAPI()


@app.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """Single prediction endpoint"""
    return await api.predict_single(request)


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(requests: List[PredictionRequest]):
    """Batch prediction endpoint"""
    return await api.predict_batch(requests)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "ensemble_loaded": api.ensemble_system.is_loaded(),
        "redis_connected": api.redis_client.ping(),
        "queue_size": api.request_queue.qsize()
    }


@app.get("/metrics")
async def get_metrics():
    """Get performance metrics"""
    return api.performance_monitor.get_metrics()