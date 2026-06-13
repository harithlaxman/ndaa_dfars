import json
import os

from openai import AzureOpenAI
from pydantic import BaseModel

OPENAI_API_KEY     = os.environ.get("AZURE_OPENAI_API_KEY")
OPENAI_ENDPOINT    = os.environ.get("AZURE_OPENAI_ENDPOINT")
OPENAI_DEPLOYMENT  = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")
OPENAI_API_VERSION = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

# OPENAI UTILS

def connect_to_openai():
    """Returns an OpenAI client"""

    if OPENAI_API_KEY is None or OPENAI_ENDPOINT is None:
        raise RuntimeError('OPENAI_API_KEY or OPENAI_ENDPOINT is not set in the environment')

    client = AzureOpenAI(
        api_version=OPENAI_API_VERSION,
        azure_endpoint=OPENAI_ENDPOINT,
        api_key=OPENAI_API_KEY
    )

    return client

def get_response(client, system_prompt: str, content: str):
    response = client.responses.create(
        model=OPENAI_DEPLOYMENT,
        input = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content
            }
        ],
    )
    return response.output_text

def get_structured_response(client, system_prompt: str, content: str, output_format: BaseModel):
    input = []
    if system_prompt != "":
        input.append({
            "role": "system",
            "content": system_prompt
        })
    input.append({
        "role": "user",
        "content": content
    })
    response = client.responses.parse(
        model=OPENAI_DEPLOYMENT,
        input = input,
        text_format=output_format
    )
    return response.output_parsed


def get_structured_response_from_input(client, input_messages, output_format: BaseModel):
    """Coerce an existing Responses conversation into a structured Pydantic object.

    Unlike get_structured_response (which takes a system prompt + a single user string),
    this accepts the full ``input`` list as built up by a tool-calling loop, so the model
    can emit its final answer with all the tool results already in context.
    """
    response = client.responses.parse(
        model=OPENAI_DEPLOYMENT,
        input=input_messages,
        text_format=output_format,
    )
    return response.output_parsed


def run_tool_loop(client, input_messages, tools, dispatch, max_turns: int = 8):
    """Drive an Azure Responses API function-calling loop until the model stops calling tools.

    Args:
        client: an AzureOpenAI client from connect_to_openai().
        input_messages: the running Responses ``input`` list (system + user messages to start).
            Mutated and returned so the caller can pass it to a final structured parse.
        tools: a list of Responses-API function tool dicts, each shaped
            ``{"type": "function", "name", "description", "parameters": <json-schema>}``.
        dispatch: maps a tool name to the Python callable that implements it. Each call is
            invoked with the model-supplied arguments as keyword args.
        max_turns: hard cap on model turns, so a misbehaving model can't loop forever.

    Returns:
        The full ``input_messages`` conversation, including every function_call the model
        made and the function_call_output we fed back. Tool errors and empty (``None``)
        results are reported back to the model as short strings rather than raised, so the
        loop is resilient.
    """
    for _ in range(max_turns):
        response = client.responses.create(
            model=OPENAI_DEPLOYMENT,
            input=input_messages,
            tools=tools,
        )
        input_messages += response.output

        calls = [item for item in response.output if getattr(item, "type", None) == "function_call"]
        if not calls:
            break

        for call in calls:
            try:
                args = json.loads(call.arguments) if call.arguments else {}
                print(f"  tool: {call.name}({json.dumps(args)})")
                result = dispatch[call.name](**args)
                output = result if isinstance(result, str) else json.dumps(result)
                if not output:
                    output = "<no result found>"
            except Exception as e:  # surface tool failures to the model, don't crash the loop
                output = f"<tool error: {e}>"
            input_messages.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": output,
                }
            )

    return input_messages