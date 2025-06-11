import logging
import multiprocessing
import os
import time
from multiprocessing import Pool
from typing import Any, Dict, List, Tuple, cast

import openai
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from financerag.common.protocols import Generator

openai.api_key = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

class OpenAIGenerator(Generator):
    """
    Customized OpenAI response generator for financial document interaction.
    Supports parallel generation with retries, GPT-4 defaults, and contextual prompting.
    """

    def __init__(self, model_name: str = "gpt-4"):
        self.model_name: str = model_name
        self.results: Dict = {}

    def _process_query(
        self, args: Tuple[str, List[ChatCompletionMessageParam], Dict[str, Any]]
    ) -> Tuple[str, str]:
        q_id, messages, kwargs = args
        temperature = kwargs.get("temperature", 0.3)
        max_tokens = kwargs.get("max_tokens", 800)
        retries = kwargs.get("retries", 2)

        # Add a system prompt if missing
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {
                "role": "system",
                "content": "You are a helpful assistant specialized in financial documents like 10-K filings and earnings reports."
            })

        client = openai.OpenAI()

        for attempt in range(retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return q_id, response.choices[0].message.content
            except Exception as e:
                logger.warning(f"[{q_id}] Retry {attempt + 1}/{retries} failed: {e}")
                time.sleep(1)

        logger.error(f"[{q_id}] Failed after {retries + 1} attempts.")
        return q_id, "Error: Unable to generate response at this time."

    def generation(
        self,
        messages: Dict[str, List[Dict[str, str]]],
        num_processes: int = multiprocessing.cpu_count(),
        **kwargs,
    ) -> Dict[str, str]:
        logger.info(
            f"Starting generation for {len(messages)} queries using {num_processes} processes..."
        )

        query_args = [
            (q_id, cast(List[ChatCompletionMessageParam], msg), kwargs.copy())
            for q_id, msg in messages.items()
        ]

        with Pool(processes=num_processes) as pool:
            results = pool.map(self._process_query, query_args)

        self.results = {q_id: content for q_id, content in results}
        logger.info(f"Generated {len(self.results)} responses successfully.")

        return self.results
