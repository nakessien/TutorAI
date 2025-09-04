import os
import time
import logging
import yaml
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import threading

try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from utils.prompts import PromptTemplates, ResponseStyle, get_next_style_in_sequence


@dataclass
class GenerationRequest:
    """LLM generation request"""
    question: str
    context: str = ""
    style: ResponseStyle = ResponseStyle.BALANCED
    user_id: str = "default"
    user_preferences: Optional[Dict[str, Any]] = None
    request_id: str = ""


@dataclass
class GenerationResponse:
    """LLM generation response"""
    content: str
    style: ResponseStyle
    generation_time: float
    token_count: int
    request_id: str
    metadata: Dict[str, Any]


class ModelStatus(Enum):
    """Model status enumeration"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    BUSY = "busy"


class LLMService:
    """LLM service with simplified implementation"""

    def __init__(self, config_or_path: Union[str, Dict[str, Any]] = "config.yaml"):
        self.logger = logging.getLogger("llm_service")

        # 支持传入配置dict或配置文件路径
        if isinstance(config_or_path, dict):
            self.config = config_or_path
            self.logger.info("LLM service initialized with config dict")
        elif isinstance(config_or_path, str):
            self.config = self._load_config(config_or_path)
            self.logger.info(f"LLM service initialized with config file: {config_or_path}")
        else:
            self.logger.error(f"Invalid config type: {type(config_or_path)}")
            self.config = self._get_default_config()

        self.prompt_templates = PromptTemplates()

        # Model management
        self.model: Optional[Llama] = None
        self.model_status = ModelStatus.UNLOADED
        self.model_lock = threading.RLock()

        # Simple caching (disabled for more authentic responses)
        self.cache_enabled = False
        self.response_cache: Dict[str, GenerationResponse] = {}

        # Performance tracking
        self.performance_stats = {
            "total_requests": 0,
            "total_generation_time": 0.0,
            "average_generation_time": 0.0,
            "errors": 0
        }

        self.logger.info("LLM service initialized successfully")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            self.logger.info(f"Successfully loaded config from {config_path}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config from {config_path}: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict[str, Any]:
        """Default configuration"""
        return {
            "llm": {
                "model_path": "./data/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
                "n_ctx": 4096,
                "n_threads": 4,
                "temperature": 0.7,
                "max_tokens": 512,
                "style_temperatures": {
                    "balanced": 0.7,
                    "detailed_policy": 0.3,
                    "practical_guide": 0.8
                }
            }
        }

    def load_model(self) -> bool:
        """Load LLM model"""
        if Llama is None:
            self.logger.error("llama-cpp-python not available")
            return False

        with self.model_lock:
            if self.model_status == ModelStatus.READY:
                self.logger.info("Model already loaded")
                return True

            if self.model_status == ModelStatus.LOADING:
                self.logger.info("Model loading in progress")
                return False

            self.model_status = ModelStatus.LOADING

            try:
                llm_config = self.config.get("llm", {})
                model_path = llm_config.get("model_path")

                if not model_path:
                    self.logger.error("Model path not specified in config")
                    self.model_status = ModelStatus.ERROR
                    return False

                if not os.path.exists(model_path):
                    self.logger.error(f"Model file not found: {model_path}")
                    self.model_status = ModelStatus.ERROR
                    return False

                self.logger.info(f"Loading model from: {model_path}")
                start_time = time.time()

                self.model = Llama(
                    model_path=model_path,
                    n_ctx=llm_config.get("n_ctx", 4096),
                    n_threads=llm_config.get("n_threads", 4),
                    n_gpu_layers=llm_config.get("n_gpu_layers", 0),
                    verbose=llm_config.get("verbose", False),
                    seed=llm_config.get("seed", -1)
                )

                load_time = time.time() - start_time
                self.model_status = ModelStatus.READY

                self.logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
                return True

            except Exception as e:
                self.model_status = ModelStatus.ERROR
                self.logger.error(f"Failed to load model: {e}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return False

    def is_model_ready(self) -> bool:
        """Check if model is ready"""
        return self.model_status == ModelStatus.READY and self.model is not None

    def generate_response(self, request: GenerationRequest) -> GenerationResponse:
        """Generate response for given request"""
        start_time = time.time()

        try:
            # Ensure model is loaded
            if not self.is_model_ready():
                self.logger.info("Model not ready, attempting to load...")
                if not self.load_model():
                    raise RuntimeError("Model failed to load")

            # Build prompt
            full_prompt = self._build_prompt(request)
            self.logger.debug(f"Generated prompt length: {len(full_prompt)}")

            # Get generation parameters
            generation_params = self._get_generation_params(request.style, request.user_preferences)

            # Generate response
            with self.model_lock:
                self.model_status = ModelStatus.BUSY

                try:
                    response_text = self._generate_with_model(full_prompt, generation_params)
                    token_count = len(response_text.split())

                finally:
                    self.model_status = ModelStatus.READY

            # Post-process response (极简化)
            processed_response = self._post_process_response(response_text)

            generation_time = time.time() - start_time

            # Create response object
            response = GenerationResponse(
                content=processed_response,
                style=request.style,
                generation_time=generation_time,
                token_count=token_count,
                request_id=request.request_id,
                metadata={
                    "prompt_length": len(full_prompt),
                    "generation_params": generation_params
                }
            )

            # Update stats
            self._update_performance_stats(generation_time)

            self.logger.info(f"Generated response for {request.request_id} in {generation_time:.2f}s")
            return response

        except Exception as e:
            self.performance_stats["errors"] += 1
            self.logger.error(f"Generation failed for {request.request_id}: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")

            return GenerationResponse(
                content=self._get_error_response(str(e)),
                style=request.style,
                generation_time=time.time() - start_time,
                token_count=0,
                request_id=request.request_id,
                metadata={"error": str(e)}
            )

    def generate_next_style_response(self,
                                     question: str,
                                     context: str,
                                     current_styles: List[str],
                                     user_id: str = "default") -> GenerationResponse:
        """Generate next style in sequence"""
        next_style_name = get_next_style_in_sequence(current_styles)
        next_style = ResponseStyle(next_style_name)

        request = GenerationRequest(
            question=question,
            context=context,
            style=next_style,
            user_id=user_id,
            request_id=f"{user_id}_{next_style_name}_{int(time.time())}"
        )

        return self.generate_response(request)

    def get_preferred_style_for_user(self, user_id: str) -> str:
        """Get preferred starting style for user (simplified)"""
        # This would integrate with preference_db in full implementation
        # For now, always start with balanced
        return "balanced"

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Build complete prompt - 使用原有的提示词模板系统"""
        try:
            return self.prompt_templates.get_complete_prompt(
                style=request.style,
                question=request.question,
                context=request.context,
                user_preferences=request.user_preferences
            )
        except Exception as e:
            self.logger.error(f"Failed to build prompt: {e}")
            return self._get_fallback_prompt(request.question, request.context)

    def _get_generation_params(self,
                               style: ResponseStyle,
                               user_preferences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get generation parameters for style"""
        llm_config = self.config.get("llm", {})

        # Base parameters
        params = {
            "max_tokens": llm_config.get("max_tokens", 512),
            "temperature": llm_config.get("style_temperatures", {}).get(
                style.value, llm_config.get("temperature", 0.7)
            ),
            "top_p": llm_config.get("top_p", 0.9),
            "top_k": llm_config.get("top_k", 40),
            "repeat_penalty": llm_config.get("repeat_penalty", 1.1),
            # 简化停止词，只保留最基本的
            "stop": ["Human:", "User:", "Question:", "\n\n\n"]
        }

        # User preference adjustments
        if user_preferences:
            if user_preferences.get("prefers_shorter_responses"):
                params["max_tokens"] = min(params["max_tokens"], 256)
            elif user_preferences.get("prefers_longer_responses"):
                params["max_tokens"] = min(int(params["max_tokens"] * 1.5), 1024)

        return params

    def _generate_with_model(self, prompt: str, params: Dict[str, Any]) -> str:
        """Generate text using model"""
        if not self.model:
            raise RuntimeError("Model not loaded")

        try:
            self.logger.debug(f"Generating with params: {params}")
            self.logger.debug(f"Prompt preview (first 500 chars): {prompt[:500]}")

            output = self.model(
                prompt,
                max_tokens=params.get("max_tokens", 512),
                temperature=params.get("temperature", 0.7),
                top_p=params.get("top_p", 0.9),
                top_k=params.get("top_k", 40),
                repeat_penalty=params.get("repeat_penalty", 1.1),
                stop=params.get("stop", []),
                echo=False
            )

            response_text = output["choices"][0]["text"]
            self.logger.debug(f"Raw model output length: {len(response_text)}")
            self.logger.debug(f"Model output preview: {response_text[:200]}")

            return response_text

        except Exception as e:
            self.logger.error(f"Model generation failed: {e}")
            raise

    def _post_process_response(self, response: str) -> str:
        """极简化的后处理 - 只做最基本的清理"""
        # 确保有响应
        if not response or len(response.strip()) < 10:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        # 基本清理：去除首尾空白
        response = response.strip()
        
        # 只移除最明显的提示词残留（如果有的话）
        obvious_prompts = ["RESPONSE:", "Response:", "Answer:", "回答："]
        for prompt in obvious_prompts:
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
                break
        
        # 确保响应不为空
        if not response:
            return "I apologize, but I couldn't generate a proper response. Please try rephrasing your question."
        
        return response

    def _update_performance_stats(self, generation_time: float):
        """Update performance statistics"""
        self.performance_stats["total_requests"] += 1
        self.performance_stats["total_generation_time"] += generation_time
        self.performance_stats["average_generation_time"] = (
                self.performance_stats["total_generation_time"] /
                self.performance_stats["total_requests"]
        )

    def _get_error_response(self, error_message: str) -> str:
        """Generate error response"""
        return f"""I apologize, but I encountered an issue while processing your request.

**What you can do:**
1. Try rephrasing your question
2. Contact the Academic Affairs Office directly
3. Check if there are any system updates in progress

Please try again in a moment, or contact technical support if the issue persists.

Technical details: {error_message}"""

    def _get_fallback_prompt(self, question: str, context: str) -> str:
        """Fallback prompt when template fails"""
        return f"""Please answer the following student question based on the provided context:

Context: {context}

Question: {question}

Please provide a helpful, accurate response."""

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        stats["model_status"] = self.model_status.value
        stats["cache_size"] = len(self.response_cache)
        return stats

    def get_model_status(self) -> str:
        """Get current model status"""
        return self.model_status.value

    def clear_cache(self):
        """Clear response cache"""
        self.response_cache.clear()
        self.logger.info("Response cache cleared")

    def shutdown(self):
        """Shutdown LLM service"""
        self.logger.info("Shutting down LLM service...")

        with self.model_lock:
            if self.model:
                del self.model
                self.model = None
            self.model_status = ModelStatus.UNLOADED

        self.clear_cache()
        self.logger.info("LLM service shutdown complete")


def create_llm_service(config_or_path: Union[str, Dict[str, Any]] = "config.yaml") -> LLMService:
    """Create LLM service instance"""
    service = LLMService(config_or_path)
    return service