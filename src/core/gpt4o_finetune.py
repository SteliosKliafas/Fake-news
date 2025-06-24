import json
import time
from typing import List, Dict
import openai  # Make sure you have the OpenAI Python SDK installed and configured

class GPT4oFinetuningPipeline:
    def __init__(self):
        self.client = openai.OpenAI()
        self.fine_tuned_models = {}

    def create_instruction_dataset(self, raw_data: List[Dict]) -> List[Dict]:
        """Create sophisticated instruction dataset for GPT-4o"""

        instruction_templates = {
            'general': self._create_general_template(),
            'chain_of_thought': self._create_cot_template(),
            'multi_perspective': self._create_multiperspective_template(),
            'evidence_based': self._create_evidence_template()
        }

        enhanced_dataset = []

        for item in raw_data:
            # Create multiple instruction variants for each sample
            for template_name, template in instruction_templates.items():
                enhanced_item = self._apply_template(item, template, template_name)
                enhanced_dataset.append(enhanced_item)

        return enhanced_dataset

    def _create_general_template(self) -> Dict:
        return {
            "system_prompt": """You are an expert fact-checker and misinformation analyst with extensive experience in journalism, digital forensics, and information verification. Your task is to analyze news articles and determine their credibility.

                                Consider these factors in your analysis:
                                1. Source credibility and verification
                                2. Factual accuracy and evidence quality  
                                3. Logical consistency and reasoning
                                4. Potential bias and emotional manipulation
                                5. Temporal plausibility and context
                                6. Cross-referencing with reliable sources
                                
                                Provide a structured analysis ending with a clear classification: REAL or FAKE, along with a confidence score (0-100).""",

                                            "user_template": """Analyze this news article for credibility:
                                
                                Article: {text}
                                
                                Provide your analysis following this structure:
                                1. Source Assessment
                                2. Content Analysis  
                                3. Evidence Evaluation
                                4. Bias Detection
                                5. Final Classification (REAL/FAKE)
                                6. Confidence Score (0-100)
                                7. Key Reasoning""",

            "assistant_template": self._generate_expert_analysis
        }

    def _create_cot_template(self) -> Dict:
        return {
            "system_prompt": """You are an expert fact-checker. Use step-by-step reasoning to analyze news articles. Think through each step of your analysis process clearly and show your reasoning chain.""",

            "user_template": """Let's analyze this news article step by step:

                                Article: {text}
                                
                                Please think through this systematically:
                                Step 1: What type of news is this?
                                Step 2: What claims are being made?
                                Step 3: What evidence is provided?
                                Step 4: Are there any red flags?
                                Step 5: What would I need to verify?
                                Step 6: Based on all factors, is this REAL or FAKE?""",

            "assistant_template": self._generate_cot_analysis
        }

    def _create_multiperspective_template(self) -> Dict:
        return {
            "system_prompt": """You are an expert fact-checker who analyzes news articles from multiple perspectives to provide a balanced and thorough assessment.""",

            "user_template": """Analyze the article from multiple perspectives:

                                Article: {text}
                                
                                Provide perspectives from:
                                1. Supportive viewpoint
                                2. Skeptical viewpoint
                                3. Neutral viewpoint
                                
                                Conclude with your overall assessment.""",

            "assistant_template": self._generate_multiperspective_analysis
        }

    def _create_evidence_template(self) -> Dict:
        return {
            "system_prompt": """You are an expert fact-checker focusing on the quality and reliability of evidence presented in news articles.""",

            "user_template": """Focus on the evidence in this article:

                                Article: {text}
                                
                                Evaluate the quality of evidence and provide classification based on this evaluation.""",

            "assistant_template": self._generate_evidence_based_analysis
        }

    def _apply_template(self, item: Dict, template: Dict, template_name: str) -> Dict:
        """Apply templates to a raw data item and generate assistant response"""

        # Format user prompt by injecting the article text
        user_prompt = template['user_template'].format(text=item.get('text', ''))

        # Generate assistant output using corresponding method
        assistant_output = template['assistant_template'](item, template_name)

        return {
            'system_prompt': template['system_prompt'],
            'user_prompt': user_prompt,
            'assistant_response': assistant_output
        }

    def _generate_expert_analysis(self, item, template_name):
        # Example expert-style fact-check analysis for training
        return (
            "1. Source Assessment: Source appears credible based on known reputation.\n"
            "2. Content Analysis: The article presents verifiable facts with minor opinions.\n"
            "3. Evidence Evaluation: Supports claims with cited documents and expert statements.\n"
            "4. Bias Detection: Minimal bias detected; balanced viewpoints included.\n"
            "5. Final Classification: REAL\n"
            "6. Confidence Score: 85\n"
            "7. Key Reasoning: Article cross-references multiple reliable sources and lacks contradictory evidence."
        )

    def _generate_cot_analysis(self, item, template_name):
        # Chain-of-thought style stepwise reasoning example
        return (
            "Step 1: This is a political news article.\n"
            "Step 2: Claims include new policy implementation.\n"
            "Step 3: Provides official government statements.\n"
            "Step 4: No immediate red flags such as anonymous sources or emotional language.\n"
            "Step 5: Verification would require official government release confirmation.\n"
            "Step 6: Given evidence, the article is REAL."
        )

    def _generate_multiperspective_analysis(self, item, template_name):
        # Multi-perspective example training response
        return (
            "Perspective 1 (Supportive): The article aligns with official reports and expert opinions.\n"
            "Perspective 2 (Skeptical): Lack of independent corroboration noted.\n"
            "Perspective 3 (Neutral): Presents both sides but omits some context.\n"
            "Final assessment: Likely REAL but requires further verification."
        )

    def _generate_evidence_based_analysis(self, item, template_name):
        # Evidence-focused analysis example
        return (
            "Evidence Quality: STRONG\n"
            "Cited studies and official data corroborate claims.\n"
            "No contradictory evidence found.\n"
            "Classification: REAL\n"
            "Confidence: 90"
        )

    def _prepare_training_file(self, training_data: List[Dict], domain: str) -> str:
        """Prepare training data in OpenAI fine-tune JSONL format"""
        filename = f"training_{domain}.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            for item in training_data:
                # Combine system + user prompts as 'prompt', assistant response as 'completion'
                prompt = f"{item['system_prompt']}\n\n{item['user_prompt']}\n\nAssistant:"
                completion = f" {item['assistant_response']}"  # note leading space per OpenAI recommendation

                json_line = json.dumps({
                    "prompt": prompt,
                    "completion": completion
                })
                f.write(json_line + "\n")
        return filename

    def _prepare_validation_file(self, validation_data: List[Dict], domain: str) -> str:
        """Prepare validation file in the same format"""
        filename = f"validation_{domain}.jsonl"
        with open(filename, 'w', encoding='utf-8') as f:
            for item in validation_data:
                prompt = f"{item['system_prompt']}\n\n{item['user_prompt']}\n\nAssistant:"
                completion = f" {item['assistant_response']}"

                json_line = json.dumps({
                    "prompt": prompt,
                    "completion": completion
                })
                f.write(json_line + "\n")
        return filename

    def fine_tune_gpt4o_advanced(self, training_data: List[Dict], domain: str = 'general'):
        """Advanced GPT-4o fine-tuning with optimization"""

        # Prepare training and validation files
        training_file = self._prepare_training_file(training_data, domain)
        validation_file = self._prepare_validation_file(training_data, domain)

        # Upload training data file
        with open(training_file, 'rb') as f:
            training_file_obj = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )

        # Upload validation data file
        with open(validation_file, 'rb') as f:
            validation_file_obj = self.client.files.create(
                file=f,
                purpose='fine-tune'
            )

        # Start fine-tuning with advanced parameters
        fine_tuning_job = self.client.fine_tuning.jobs.create(
            training_file=training_file_obj.id,
            validation_file=validation_file_obj.id,
            model="gpt-4o-mini-2024-07-18",
            suffix=f"fake-news-{domain}",
            hyperparameters={
                "n_epochs": 3,
                "batch_size": 16,
                "learning_rate_multiplier": 0.1,
                "prompt_loss_weight": 0.01
            },
            integrations=[{
                "type": "wandb",
                "wandb": {
                    "project": "gpt4o-fake-news-detection",
                    "tags": [domain, "advanced", "ensemble"]
                }
            }]
        )

        # Monitor training progress
        self._monitor_training_progress(fine_tuning_job.id)

        # Save fine-tuned model ID for domain
        self.fine_tuned_models[domain] = fine_tuning_job.fine_tuned_model

        return fine_tuning_job

    def _monitor_training_progress(self, job_id: str):
        print(f"Monitoring fine-tuning job {job_id}...")

        while True:
            job_status = self.client.fine_tuning.jobs.retrieve(id=job_id)
            status = job_status.status

            print(f"Job status: {status}")

            if status in ("succeeded", "failed", "cancelled"):
                print(f"Fine-tuning job ended with status: {status}")
                break

            time.sleep(10)  # Poll every 10 seconds

    async def predict_with_gpt4o(self, text: str, domain: str = 'general') -> Dict:
        """Advanced GPT-4o prediction with structured output"""

        model_id = self.fine_tuned_models.get(domain, 'gpt-4o-mini')

        system_prompt = """You are an expert fact-checker. Analyze the given article and respond with a structured JSON output containing your analysis and classification."""

        user_prompt = f"""Analyze this article: {text}

                        Respond with JSON in this exact format:
                        {{
                            "classification": "REAL" or "FAKE",
                            "confidence": 0-100,
                            "reasoning": "Brief explanation of your reasoning",
                            "evidence_quality": "STRONG/MODERATE/WEAK/NONE",
                            "source_credibility": "HIGH/MEDIUM/LOW/UNKNOWN",
                            "logical_consistency": "CONSISTENT/INCONSISTENT",
                            "bias_indicators": ["list", "of", "bias", "indicators"],
                            "fact_claims": ["list", "of", "verifiable", "claims"],
                            "red_flags": ["list", "of", "warning", "signs"]
                        }}"""

        try:
            response = await self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=1000,
                response_format={"type": "json_object"}
            )

            result = json.loads(response.choices[0].message.content)

            return {
                'prediction': result['classification'],
                'confidence': result['confidence'] / 100,
                'detailed_analysis': result,
                'model_used': model_id,
                'tokens_used': response.usage.total_tokens
            }

        except Exception as e:
            return {
                'prediction': 'UNKNOWN',
                'confidence': 0.0,
                'error': str(e),
                'model_used': model_id
            }
