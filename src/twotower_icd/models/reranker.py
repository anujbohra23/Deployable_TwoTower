"""
LLM-based re-ranker for retrieved ICD codes.
"""
import torch
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json


@dataclass
class RerankResult:
    """Result from re-ranking"""
    patient_id: str
    original_scores: List[Tuple[str, float]]  # (code, retrieval_score)
    reranked_codes: List[str]  # Final ordered list
    reranked_scores: List[float]  # LLM confidence scores
    llm_reasoning: str  # Optional explanation


class LLMReranker:
    """
    Re-ranks retrieved ICD codes using a free LLM API.
    Supports multiple backends: OpenRouter (free models), HuggingFace Inference API, etc.
    """
    
    def __init__(
        self,
        backend: str = "openrouter",  # "openrouter", "huggingface", "ollama"
        model_name: str = "meta-llama/llama-3.2-3b-instruct:free",
        api_key: str = None,
        temperature: float = 0.1,
        max_tokens: int = 2000
    ):
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Configure API endpoint
        if backend == "openrouter":
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        elif backend == "huggingface":
            self.api_url = f"https://api-inference.huggingface.co/models/{model_name}"
        elif backend == "ollama":
            self.api_url = "http://localhost:11434/api/generate"
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def build_rerank_prompt(
        self,
        patient_text: str,
        patient_demographics: Dict,
        patient_labs: Dict,
        candidate_codes: List[Tuple[str, str, float]],  # (code, description, score)
        top_n: int = 20
    ) -> str:
        """
        Constructs the prompt for LLM re-ranking.
        """
        
        prompt = f"""You are a medical coding expert. Your task is to re-rank ICD-10 diagnosis codes for a patient based on their clinical information.

**PATIENT INFORMATION:**

Demographics:
- Age: {patient_demographics.get('age', 'N/A')}
- Sex: {patient_demographics.get('sex', 'N/A')}

Laboratory Values:
"""
        
        # Add lab values
        for lab_name, lab_value in patient_labs.items():
            if lab_value is not None:
                prompt += f"- {lab_name}: {lab_value}\n"
        
        prompt += f"""
Clinical Notes:
{patient_text[:2000]}  # Truncate to avoid token limits

---

**CANDIDATE ICD-10 CODES** (retrieved by similarity search):

"""
        
        # Add candidate codes
        for idx, (code, description, score) in enumerate(candidate_codes, 1):
            prompt += f"{idx}. **{code}**: {description} (retrieval score: {score:.3f})\n"
        
        prompt += f"""
---

**TASK:**
Analyze the patient's clinical information and re-rank these {len(candidate_codes)} ICD-10 codes by relevance. 
Select and rank the top {top_n} most appropriate codes based on:
1. Direct mention or strong clinical evidence in the notes
2. Correlation with laboratory findings
3. Coherence with demographics (age/sex)
4. Clinical likelihood and diagnostic guidelines

**OUTPUT FORMAT (JSON only, no additional text):**
```json
{{
  "reranked_codes": [
    {{"code": "E11.9", "rank": 1, "confidence": 0.95, "reasoning": "Strong evidence of diabetes in notes and lab values"}},
    {{"code": "I10", "rank": 2, "confidence": 0.87, "reasoning": "Hypertension mentioned explicitly"}},
    ...
  ]
}}
```

Respond with JSON only.
"""
        return prompt
    
    def call_llm(self, prompt: str) -> str:
        """
        Calls the LLM API and returns the response.
        """
        
        if self.backend == "openrouter":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a medical coding expert specializing in ICD-10 diagnosis codes."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
        
        elif self.backend == "huggingface":
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": self.max_tokens,
                    "return_full_text": False
                }
            }
            
            response = requests.post(self.api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result[0]["generated_text"]
        
        elif self.backend == "ollama":
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature
                }
            }
            
            response = requests.post(self.api_url, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            return result["response"]
        
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
    
    def parse_llm_response(self, response: str) -> List[Dict]:
        """
        Parses the LLM's JSON response into a structured format.
        """
        try:
            # Extract JSON from markdown code blocks if present
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            parsed = json.loads(response)
            return parsed["reranked_codes"]
        
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Response was: {response}")
            return []
    
    def rerank(
        self,
        patient_id: str,
        patient_text: str,
        patient_demographics: Dict,
        patient_labs: Dict,
        candidate_codes: List[Tuple[str, str, float]],  # (code, description, retrieval_score)
        top_n: int = 20
    ) -> RerankResult:
        """
        Re-ranks candidate ICD codes using LLM.
        
        Args:
            patient_id: Patient identifier
            patient_text: Clinical notes
            patient_demographics: Dict with age, sex, etc.
            patient_labs: Dict with lab values
            candidate_codes: List of (code, description, score) tuples from retrieval
            top_n: Number of final codes to return
        
        Returns:
            RerankResult object with reranked codes
        """
        
        # Build prompt
        prompt = self.build_rerank_prompt(
            patient_text,
            patient_demographics,
            patient_labs,
            candidate_codes,
            top_n
        )
        
        # Call LLM
        try:
            llm_response = self.call_llm(prompt)
            
            # Parse response
            reranked_items = self.parse_llm_response(llm_response)
            
            if not reranked_items:
                # Fallback: return original ranking
                print(f"Warning: LLM parsing failed for patient {patient_id}, using retrieval ranking")
                return RerankResult(
                    patient_id=patient_id,
                    original_scores=[(code, score) for code, _, score in candidate_codes],
                    reranked_codes=[code for code, _, _ in candidate_codes[:top_n]],
                    reranked_scores=[score for _, _, score in candidate_codes[:top_n]],
                    llm_reasoning="Fallback to retrieval ranking"
                )
            
            # Extract reranked codes and scores
            reranked_codes = [item["code"] for item in reranked_items[:top_n]]
            reranked_scores = [item.get("confidence", 0.0) for item in reranked_items[:top_n]]
            
            return RerankResult(
                patient_id=patient_id,
                original_scores=[(code, score) for code, _, score in candidate_codes],
                reranked_codes=reranked_codes,
                reranked_scores=reranked_scores,
                llm_reasoning=llm_response
            )
        
        except Exception as e:
            print(f"Error during re-ranking for patient {patient_id}: {e}")
            # Fallback to retrieval ranking
            return RerankResult(
                patient_id=patient_id,
                original_scores=[(code, score) for code, _, score in candidate_codes],
                reranked_codes=[code for code, _, _ in candidate_codes[:top_n]],
                reranked_scores=[score for _, _, score in candidate_codes[:top_n]],
                llm_reasoning=f"Error: {str(e)}"
            )
    
    def batch_rerank(
        self,
        patient_data: List[Dict],
        candidate_codes_batch: List[List[Tuple[str, str, float]]],
        top_n: int = 20
    ) -> List[RerankResult]:
        """
        Re-ranks multiple patients sequentially.
        
        Args:
            patient_data: List of dicts with patient info
            candidate_codes_batch: List of candidate code lists
            top_n: Number of final codes per patient
        
        Returns:
            List of RerankResult objects
        """
        results = []
        
        for patient_info, candidate_codes in zip(patient_data, candidate_codes_batch):
            result = self.rerank(
                patient_id=patient_info["patient_id"],
                patient_text=patient_info["text"],
                patient_demographics=patient_info["demographics"],
                patient_labs=patient_info["labs"],
                candidate_codes=candidate_codes,
                top_n=top_n
            )
            results.append(result)
        
        return results