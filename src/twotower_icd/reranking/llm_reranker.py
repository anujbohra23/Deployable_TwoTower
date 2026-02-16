"""
LLM-based reranker for ICD code retrieval.
Uses free LLM APIs to rerank retrieved ICD codes based on patient context.
"""

from __future__ import annotations
import os
import json
from typing import Dict, List, Optional
import requests
from enum import Enum


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    GROQ = "groq"
    TOGETHER = "together"
    OPENROUTER = "openrouter"
    LOCAL = "local"


class LLMReranker:
    """
    Reranks ICD codes using LLM-based relevance scoring.
    
    Supports multiple free LLM providers including OpenRouter.
    """
    
    def __init__(
        self,
        provider: str = "huggingface",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        max_candidates: int = 20,
    ):
        """
        Initialize the LLM reranker.
        
        Args:
            provider: LLM provider ('huggingface', 'openai', 'groq', 'together', 'openrouter')
            model_name: Model name (provider-specific)
            api_key: API key for the provider (if required)
            max_candidates: Maximum number of candidates to rerank
        """
        self.provider = LLMProvider(provider.lower())
        self.max_candidates = max_candidates
        
        # Set default models per provider
        if model_name is None:
            model_name = self._get_default_model()
        
        self.model_name = model_name
        
        # Get API key from environment or parameter
        self.api_key = api_key or self._get_api_key()
        
        # Provider-specific configuration
        self._setup_provider()
    
    def _get_default_model(self) -> str:
        """Get default model name for the provider."""
        defaults = {
            LLMProvider.HUGGINGFACE: "mistralai/Mistral-7B-Instruct-v0.2",
            LLMProvider.OPENAI: "gpt-3.5-turbo",
            LLMProvider.GROQ: "mixtral-8x7b-32768",
            LLMProvider.TOGETHER: "mistralai/Mixtral-8x7B-Instruct-v0.1",
            LLMProvider.OPENROUTER: "meta-llama/llama-3.2-3b-instruct:free",
        }
        return defaults.get(self.provider, defaults[LLMProvider.HUGGINGFACE])
    
    def _get_api_key(self) -> Optional[str]:
        """Get API key from environment variables."""
        env_vars = {
            LLMProvider.HUGGINGFACE: "HUGGINGFACE_API_KEY",
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.TOGETHER: "TOGETHER_API_KEY",
            LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
        }
        env_var = env_vars.get(self.provider)
        return os.getenv(env_var) if env_var else None
    
    def _setup_provider(self):
        """Setup provider-specific URLs and headers."""
        if self.provider == LLMProvider.HUGGINGFACE:
            self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            self.headers = {}
            if self.api_key:
                self.headers["Authorization"] = f"Bearer {self.api_key}"
        elif self.provider == LLMProvider.OPENAI:
            self.api_url = "https://api.openai.com/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        elif self.provider == LLMProvider.GROQ:
            self.api_url = "https://api.groq.com/openai/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        elif self.provider == LLMProvider.TOGETHER:
            self.api_url = "https://api.together.xyz/inference"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        elif self.provider == LLMProvider.OPENROUTER:
            self.api_url = "https://openrouter.ai/api/v1/chat/completions"
            self.headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/twotower-icd",  # Optional but recommended
            }
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _build_prompt(
        self,
        clinical_note: str,
        lab_summary: str,
        age: float,
        sex: str,
        candidate_codes: List[Dict[str, str]],
    ) -> str:
        """Build prompt for LLM reranking."""
        codes_text = "\n".join([
            f"{i+1}. [{code['code']}] {code['title']}\n   Description: {code['description'][:200]}"
            for i, code in enumerate(candidate_codes)
        ])
        
        prompt = f"""You are a medical coding expert. Given a patient's clinical information and a list of candidate ICD-10 codes, rank the codes by relevance to the patient's condition.

Patient Information:
- Age: {age}
- Sex: {sex}
- Clinical Note: {clinical_note[:1500]}
- Lab Summary: {lab_summary}

Candidate ICD-10 Codes:
{codes_text}

Please rank these codes from most relevant (1) to least relevant ({len(candidate_codes)}). Return ONLY a JSON array of code numbers in ranked order, like: [3, 1, 5, 2, 4, ...]

Ranked order (JSON array):"""
        
        return prompt
    
    def _call_huggingface(self, prompt: str) -> str:
        """Call Hugging Face Inference API."""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 200,
                "temperature": 0.1,
                "return_full_text": False,
            }
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Hugging Face API error: {response.status_code} - {response.text}")
        
        result = response.json()
        
        # Handle different response formats
        if isinstance(result, list) and len(result) > 0:
            if isinstance(result[0], dict) and "generated_text" in result[0]:
                return result[0]["generated_text"]
            elif isinstance(result[0], str):
                return result[0]
        elif isinstance(result, dict):
            if "generated_text" in result:
                return result["generated_text"]
            elif "text" in result:
                return result["text"]
        
        return str(result)
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a medical coding expert. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenAI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API (OpenAI-compatible)."""
        return self._call_openai(prompt)  # Same format as OpenAI
    
    def _call_openrouter(self, prompt: str) -> str:
        """Call OpenRouter API (OpenAI-compatible)."""
        # OpenRouter uses OpenAI-compatible format
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a medical coding expert. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200,
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"OpenRouter API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def _call_together(self, prompt: str) -> str:
        """Call Together AI API."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": 200,
            "temperature": 0.1,
        }
        
        response = requests.post(
            self.api_url,
            headers=self.headers,
            json=payload,
            timeout=30,
        )
        
        if response.status_code != 200:
            raise RuntimeError(f"Together AI API error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["output"]["choices"][0]["text"]
    
    def _parse_ranking(self, llm_output: str, num_candidates: int) -> List[int]:
        """Parse LLM output to extract ranked indices."""
        # Try to extract JSON array
        try:
            import re
            json_match = re.search(r'\[[\d\s,]+\]', llm_output)
            if json_match:
                ranking = json.loads(json_match.group())
                # Convert to 0-indexed and validate
                ranking = [int(x) - 1 for x in ranking if 0 <= int(x) - 1 < num_candidates]
                if len(ranking) == num_candidates:
                    return ranking
        except:
            pass
        
        # Fallback: try to extract numbers
        import re
        numbers = [int(x) - 1 for x in re.findall(r'\b(\d+)\b', llm_output) 
                  if 0 <= int(x) - 1 < num_candidates]
        
        if len(numbers) >= num_candidates:
            return numbers[:num_candidates]
        
        # Ultimate fallback: return original order
        return list(range(num_candidates))
    
    def rerank(
        self,
        candidate_codes: List[Dict[str, any]],
        clinical_note: str,
        lab_values: Dict[str, float],
        age: float,
        sex: str,
    ) -> List[Dict[str, any]]:
        """
        Rerank candidate ICD codes using LLM.
        
        Args:
            candidate_codes: List of candidate codes with 'code', 'title', 'description', 'score'
            clinical_note: Patient's clinical notes
            lab_values: Dict of lab values
            age: Patient age
            sex: Patient sex
            
        Returns:
            Reranked list of codes with updated 'rank' and 'rerank_score' fields
        """
        if not candidate_codes:
            return []
        
        # Limit candidates for reranking
        candidates = candidate_codes[:self.max_candidates]
        
        # Build lab summary
        lab_summary = ", ".join([
            f"{k.upper()}: {v:.2f}" for k, v in lab_values.items() if v is not None
        ]) or "No lab values available"
        
        # Build prompt
        prompt = self._build_prompt(clinical_note, lab_summary, age, sex, candidates)
        
        # Call LLM
        try:
            if self.provider == LLMProvider.HUGGINGFACE:
                llm_output = self._call_huggingface(prompt)
            elif self.provider == LLMProvider.OPENAI:
                llm_output = self._call_openai(prompt)
            elif self.provider == LLMProvider.GROQ:
                llm_output = self._call_groq(prompt)
            elif self.provider == LLMProvider.TOGETHER:
                llm_output = self._call_together(prompt)
            elif self.provider == LLMProvider.OPENROUTER:
                llm_output = self._call_openrouter(prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Parse ranking
            ranked_indices = self._parse_ranking(llm_output, len(candidates))
            
            # Reorder candidates
            reranked = [candidates[i] for i in ranked_indices]
            
            # Update ranks and add rerank flag
            for i, code in enumerate(reranked):
                code['rank'] = i + 1
                code['reranked'] = True
                code['original_rank'] = code.get('rank', i + 1)
            
            # Append any remaining candidates that weren't reranked
            if len(candidate_codes) > self.max_candidates:
                remaining = candidate_codes[self.max_candidates:]
                for i, code in enumerate(remaining):
                    code['rank'] = len(reranked) + i + 1
                    code['reranked'] = False
                reranked.extend(remaining)
            
            return reranked
            
        except Exception as e:
            # Fallback: return original order if reranking fails
            print(f"Warning: Reranking failed ({e}), returning original order")
            for i, code in enumerate(candidate_codes):
                code['rank'] = i + 1
                code['reranked'] = False
            return candidate_codes
