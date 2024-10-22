import json
import logging
import os
import sys
import warnings
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import requests
from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import (
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessageChunk,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
    pre_init,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

argo_chat_url = "https://apps.inside.anl.gov/argoapi/api/v1/resource/chat/"


class ChatArgo(BaseChatModel):
    """A custom chat model that echoes the first `n` characters of the input.

    When contributing an implementation to LangChain, carefully document
    the model including the initialization parameters, include
    an example of how to initialize the model and include any relevant
    links to the underlying models documentation or API.
    """

    user_name: str = Field(default="heng.ma", description="Your ANL usrname")
    model_name: str = Field(default="gpt4o", alias="model")
    """Model name to use."""
    temperature: float = 0.7
    """What sampling temperature to use."""
    n: int = 1
    """Number of chat completions to generate for each prompt."""
    max_tokens: int = Field(default=5000)

    model_config = ConfigDict(
        populate_by_name=True,
    )

    def _build_call(
        self,
        prompt: str,
        **kwargs: Any,
    ):
        # Data to be sent as a POST in JSON format
        data = {
            "user": self.user_name,
            "model": self.model_name,
            "system": "You are a large language model with the name Argo.",
            "prompt": [prompt],
            "stop": [],
            "temperature": 0.1,
            "top_p": 0.9,
        }
        return data

    def _generate(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Run the LLM on the given input.

        Override this method to implement the LLM logic.

        Args:
            prompt: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string. Actual completions SHOULD NOT include the prompt.
        """
        data = self._build_call(prompt)
        headers = {"Content-Type": "application/json"}
        payload = json.dumps(data)
        response = requests.post(argo_chat_url, data=payload, headers=headers)
        return response.json()

    # def _stream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[CallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> Iterator[ChatGenerationChunk]:
    #     message_dicts, params = self._create_message_dicts(messages, stop)
    #     params = {**params, **kwargs, "stream": True}

    #     default_chunk_class = AIMessageChunk
    #     for chunk in self.completion_with_retry(
    #         messages=message_dicts, run_manager=run_manager, **params
    #     ):
    #         if not isinstance(chunk, dict):
    #             chunk = chunk.dict()
    #         if len(chunk["choices"]) == 0:
    #             continue
    #         choice = chunk["choices"][0]
    #         if choice["delta"] is None:
    #             continue
    #         chunk = _convert_delta_to_message_chunk(
    #             choice["delta"], default_chunk_class
    #         )
    #         finish_reason = choice.get("finish_reason")
    #         generation_info = (
    #             dict(finish_reason=finish_reason) if finish_reason is not None else None
    #         )
    #         default_chunk_class = chunk.__class__
    #         cg_chunk = ChatGenerationChunk(
    #             message=chunk, generation_info=generation_info
    #         )
    #         if run_manager:
    #             run_manager.on_llm_new_token(cg_chunk.text, chunk=cg_chunk)
    #         yield cg_chunk

    # @property
    # def _identifying_params(self) -> Dict[str, Any]:
    #     """Return a dictionary of identifying parameters."""
    #     return {
    #         # The model name allows users to specify custom token counting
    #         # rules in LLM monitoring applications (e.g., in LangSmith users
    #         # can provide per token pricing for their model and monitor
    #         # costs for the given LLM.)
    #         "model_name": "ArgoChatModel",
    #     }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "ArgoChatModel"
