import os
import requests
import urllib3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = FastAPI(title="WSO2 Platform-Hosted HR Agent")

# Variables inyectadas automáticamente por WSO2 AMP en runtime
LLM_BASE_URL = os.environ.get("LLM_URL", "").rstrip("/")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")

MCP_URL = os.environ.get("MCP_URL", "")
MCP_API_KEY = os.environ.get("MCP_API_KEY", "")


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None
    context: dict | None = None


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not LLM_BASE_URL or not LLM_API_KEY:
        raise HTTPException(status_code=500, detail="Faltan variables LLM_URL o LLM_API_KEY")

    llm_endpoint = f"{LLM_BASE_URL}/v1beta/models/{LLM_MODEL}:generateContent"
    headers_llm = {
        "api-key": LLM_API_KEY,
        "Content-Type": "application/json",
    }

    tools = [{
        "function_declarations": [{
            "name": "getEmployeeInfo",
            "description": "Obtiene la información de perfil y cargo de un empleado del sistema de RRHH.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "employee_id": {"type": "STRING", "description": "ID del empleado"}
                },
                "required": ["employee_id"],
            },
        }]
    }]

    payload_inicial = {
        "contents": [{"role": "user", "parts": [{"text": req.message}]}],
        "tools": tools,
    }

    res_1 = requests.post(llm_endpoint, headers=headers_llm, json=payload_inicial, verify=False, timeout=30)
    if res_1.status_code != 200:
        raise HTTPException(status_code=res_1.status_code, detail=f"Error LLM Gateway: {res_1.text}")

    data_1 = res_1.json()
    part = data_1["candidates"][0]["content"]["parts"][0]

    # Si Gemini solicita invocar la herramienta MCP
    if "functionCall" in part:
        if not MCP_URL or not MCP_API_KEY:
            raise HTTPException(status_code=500, detail="Faltan variables MCP_URL o MCP_API_KEY")

        func_call = part["functionCall"]
        tool_name = func_call["name"]
        tool_args = func_call.get("args", {})
        call_id = func_call.get("id")

        headers_mcp = {
            "X-API-Key": MCP_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        mcp_payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": tool_args},
        }

        mcp_res = requests.post(MCP_URL, headers=headers_mcp, json=mcp_payload, verify=False, timeout=30)
        if mcp_res.status_code != 200:
            raise HTTPException(status_code=mcp_res.status_code, detail=f"Error MCP Proxy: {mcp_res.text}")

        mcp_data = mcp_res.json()
        hr_data = mcp_data["result"]["content"][0]["text"]

        function_response_part = {
            "functionResponse": {
                "name": tool_name,
                "response": {"result": hr_data},
            }
        }
        if call_id:
            function_response_part["functionResponse"]["id"] = call_id

        payload_final = {
            "contents": [
                {"role": "user", "parts": [{"text": req.message}]},
                data_1["candidates"][0]["content"],
                {"role": "user", "parts": [function_response_part]},
            ],
            "tools": tools,
        }

        res_2 = requests.post(llm_endpoint, headers=headers_llm, json=payload_final, verify=False, timeout=30)
        if res_2.status_code != 200:
            raise HTTPException(status_code=res_2.status_code, detail=f"Error LLM final: {res_2.text}")

        final_text = res_2.json()["candidates"][0]["content"]["parts"][0]["text"]
        return ChatResponse(response=final_text)

    return ChatResponse(response=part.get("text", ""))