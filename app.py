import secrets
import time
import uuid
import hashlib
import json
import httpx
import logging
from typing import AsyncGenerator, List, Dict, Union
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import StreamingResponse
from collections import OrderedDict
from datetime import datetime
import random,uvicorn

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# 配置
class Config(BaseModel):
    # API 密钥
    API_KEY: str = Field(
        default="sk_gUXNcLwm0rnnEt55Mg8hq88",
        description="API key for authentication"
    )
    
    # 最大历史记录数
    MAX_HISTORY: int = Field(
        default=30,
        description="Maximum number of conversation histories to keep"
    )
    
    # API 域名
    API_DOMAIN: str = Field(
        default="https://ai-api.dangbei.net",
        description="API Domain for requests"
    )
    
    # User Agents 列表
    USER_AGENTS: List[str] = Field(
        default=[
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            "Mozilla/5.0 (iPad; CPU OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"
        ],
        description="List of User Agent strings for requests"
    )

    # 每个设备 ID 最大会话数
    DEVICE_CONVERSATIONS_LIMIT: int = Field(
        default=10,
        description="Number of conversations before generating new device ID"
    )

# 创建全局配置实例
config = Config()

# 辅助函数：验证 API 密钥
async def verify_api_key(authorization: str = Header(None)):
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing API key")
    
    api_key = authorization.replace("Bearer ", "").strip()
    if api_key != config.API_KEY:  # 使用配置中的 API_KEY
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

class Message(BaseModel):
    role: str
    content: str
    
    class Config:
        # 允许额外的字段
        extra = "allow"

class ChatRequest(BaseModel):
    model: str
    messages: List[Union[dict, Message]]  # 允许字典或 Message 对象
    stream: bool = False
    
    # 添加额外的可选字段，以适应更多的客户端请求
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    user: str | None = None
    
    class Config:
        # 允许额外的字段
        extra = "allow"
        # 允许从字典直接构造
        arbitrary_types_allowed = True

    @property
    def messages_as_dicts(self) -> List[dict]:
        """将消息转换为字典格式"""
        return [
            msg if isinstance(msg, dict) else msg.dict()
            for msg in self.messages
        ]

class ChatHistory:
    def __init__(self):
        self.current_device_id = None
        self.current_conversation_id = None
        self.conversation_count = 0
        self.total_conversations = 0  # 添加总会话计数

    def get_or_create_ids(self, force_new=False) -> tuple[str, str]:
        """
        获取或创建新的 device_id 和 conversation_id
        
        Args:
            force_new (bool): 是否强制创建新会话，用于清除上下文
            
        Returns:
            tuple[str, str]: (device_id, conversation_id)
        """
        # 检查是否需要创建新的设备 ID
        if (not self.current_device_id or 
            self.total_conversations >= config.DEVICE_CONVERSATIONS_LIMIT):
            self.current_device_id = self._generate_device_id()
            self.current_conversation_id = None
            self.conversation_count = 0
            self.total_conversations = 0
            logger.info(f"Generated new device ID: {self.current_device_id}")

        # 如果强制新建会话（清除上下文）或没有当前会话 ID
        if force_new or not self.current_conversation_id:
            self.current_conversation_id = None
            self.conversation_count = 0
            logger.info("Forcing new conversation")

        return self.current_device_id, self.current_conversation_id

    def add_conversation(self, conversation_id: str):
        """
        添加新的对话记录
        
        Args:
            conversation_id (str): 新的会话 ID
        """
        if not self.current_device_id:
            return
            
        self.current_conversation_id = conversation_id
        self.conversation_count += 1
        self.total_conversations += 1
        logger.info(f"Added conversation {conversation_id} (count: {self.conversation_count}, total: {self.total_conversations})")

    def _generate_device_id(self) -> str:
        """生成新的设备ID，并随机选择新的 USER_AGENT"""
        # 随机选择新的 USER_AGENT
        user_agent = random.choice(config.USER_AGENTS)
        logger.info(f"Selected new User-Agent: {user_agent}")
        
        uuid_str = uuid.uuid4().hex
        nanoid_str = ''.join(random.choices(
            "useandom26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict",
            k=20
        ))
        return f"{uuid_str}_{nanoid_str}"

class Pipe:
    def __init__(self):
        self.data_prefix = "data:"
        self.user_agent = random.choice(config.USER_AGENTS)  # 初始化时随机选择一个 USER_AGENT
        self.chat_history = ChatHistory()
        # 添加支持联网的模型映射，保持实际请求时使用小写
        self.search_models = {
            "DeepSeek-R1-Search": "deepseek",
            "DeepSeek-V3-Search": "deepseek",
            "Doubao-Search": "doubao",    # 显示用大写，映射用小写
            "Qwen-Search": "qwen"         # 显示用大写，映射用小写
        }

    def _build_full_prompt(self, messages: List[Dict]) -> str:
        """构建完整的提示，包含系统提示、聊天历史和当前问题"""
        if not messages:
            return ''
            
        system_prompt = ''
        history = []
        last_user_message = ''
        
        # 修改消息处理逻辑，直接使用字典访问
        for msg in messages:
            if msg['role'] == 'system' and not system_prompt:
                system_prompt = msg['content']
            elif msg['role'] == 'user':
                history.append(f"user: {msg['content']}")
                last_user_message = msg['content']
            elif msg['role'] == 'assistant':
                history.append(f"assistant: {msg['content']}")
        
        # 构建最终提示
        parts = []
        if system_prompt:
            parts.append(f"[System Prompt]\n{system_prompt}")
        if len(history) > 1:  # 如果有历史对话
            parts.append(f"[Chat History]\n{chr(10).join(history[:-1])}")
        parts.append(f"[Question]\n{last_user_message}")
        
        return chr(10).join(parts)

    async def pipe(self, body: dict) -> AsyncGenerator[Dict, None]:
        thinking_state = {"thinking": -1}
        
        try:
            # 构建完整提示
            full_prompt = self._build_full_prompt(body["messages"])
            
            # 修改 force_new_context 的判断逻辑
            force_new_context = False
            messages = body["messages"]
            if len(messages) == 1:  # 只有一条消息时，说明是新对话
                force_new_context = True
            elif len(messages) >= 2:  # 检查是否清除了历史
                last_two = messages[-2:]
                if last_two[0]["role"] == "user" and last_two[1]["role"] == "user":
                    force_new_context = True
            
            # 获取或创建设备ID和会话ID
            device_id, conversation_id = self.chat_history.get_or_create_ids(force_new_context)
            
            # 添加会话信息日志
            logger.info(f"Current session - Device ID: {device_id}, Conversation ID: {conversation_id}, Force new: {force_new_context}, Messages count: {len(messages)}")

            # 如果没有会话ID，创建新的会话
            if not conversation_id:
                conversation_id = await self._create_conversation(device_id)
                if not conversation_id:
                    yield {"error": "Failed to create conversation"}
                    return
                # 保存新的对话记录
                self.chat_history.add_conversation(conversation_id)
                logger.info(f"Created new conversation: {conversation_id}")

            # 模型名称处理
            model_name = None
            is_search_model = body["model"].endswith("-Search")
            if is_search_model:
                # 如果是搜索模型，使用映射的基础模型名
                base_model = body["model"].replace("-Search", "")
                model_name = self.search_models.get(body["model"], base_model.lower())
            else:
                # 非搜索模型使用原有逻辑
                is_deepseek_model = body["model"] in ["DeepSeek-R1", "DeepSeek-V3"]
                model_name = "deepseek" if is_deepseek_model else body["model"]

            # 确定 userAction 参数
            user_action = ""
            if "DeepSeek-R1" in body["model"]:
                user_action = "deep"
            if is_search_model:
                # 如果已经有值，添加逗号分隔
                if user_action:
                    user_action += ",online"
                else:
                    user_action = "online"  # 为联网模型设置 userAction 为 "online"

            payload = {
                "stream": True,
                "botCode": "AI_SEARCH",
                "userAction": user_action,
                "model": model_name,
                "conversationId": conversation_id,
                "question": full_prompt,
            }

            timestamp = str(int(time.time()))
            nonce = self._nanoid(21)
            sign = self._generate_sign(timestamp, payload, nonce)

            headers = {
                "Origin": "https://ai.dangbei.com",
                "Referer": "https://ai.dangbei.com/",
                "User-Agent": self.user_agent,
                "deviceId": device_id,
                "nonce": nonce,
                "sign": sign,
                "timestamp": timestamp,
            }

            api = f"{config.API_DOMAIN}/ai-search/chatApi/v1/chat"  # 使用配置中的 API_DOMAIN

            async with httpx.AsyncClient() as client:
                async with client.stream("POST", api, json=payload, headers=headers, timeout=1200) as response:
                    if response.status_code != 200:
                        error = await response.aread()
                        yield {"error": self._format_error(response.status_code, error)}
                        return

                    card_messages = []  # 用于收集卡片消息
                    
                    async for line in response.aiter_lines():
                        if not line.startswith(self.data_prefix):
                            continue

                        json_str = line[len(self.data_prefix):]

                        try:
                            data = json.loads(json_str)
                        except json.JSONDecodeError as e:
                            yield {"error": f"JSONDecodeError: {str(e)}", "data": json_str}
                            return

                        if data.get("type") == "answer":
                            content = data.get("content")
                            content_type = data.get("content_type")
                            
                            # 处理思考状态
                            if thinking_state["thinking"] == -1 and content_type == "thinking":
                                thinking_state["thinking"] = 0
                                yield {"choices": [{"delta": {"content": "<think>\n\n"}, "finish_reason": None}]}
                            elif thinking_state["thinking"] == 0 and content_type == "text":
                                thinking_state["thinking"] = 1
                                yield {"choices": [{"delta": {"content": "\n"}, "finish_reason": None}]}
                                yield {"choices": [{"delta": {"content": "</think>"}, "finish_reason": None}]}
                                yield {"choices": [{"delta": {"content": "\n\n"}, "finish_reason": None}]}
                            
                            # 处理卡片内容
                            if content_type == "card":
                                try:
                                    card_content = json.loads(content)
                                    card_items = card_content["cardInfo"]["cardItems"]
                                    markdown_output = "\n\n---\n\n"
                                    
                                    # 处理搜索关键词（type: 2001）
                                    search_keywords = next((item for item in card_items if item["type"] == "2001"), None)
                                    if search_keywords:
                                        keywords = json.loads(search_keywords["content"])
                                        markdown_output += f"搜索关键字：{'; '.join(keywords)}\n"
                                    
                                    # 处理搜索结果（type: 2002）
                                    search_results = next((item for item in card_items if item["type"] == "2002"), None)
                                    if search_results:
                                        results = json.loads(search_results["content"])
                                        markdown_output += f"共找到 {len(results)} 个搜索结果：\n"
                                        for result in results:
                                            markdown_output += f"[{result['idIndex']}] [{result['name']}]({result['url']}) 来源：{result['siteName']}\n"
                                    
                                    card_messages.append(markdown_output)
                                except Exception as e:
                                    logger.error(f"Error processing card: {str(e)}")

                            # 处理普通文本内容
                            if content and content_type in ["text", "thinking"]:
                                yield {"choices": [{"delta": {"content": content}, "finish_reason": None}]}

                    # 在最后输出所有卡片消息
                    if card_messages:
                        yield {"choices": [{"delta": {"content": "".join(card_messages)}, "finish_reason": None}]}

                    # 在最后添加元数据
                    yield {"choices": [{"delta": {"meta": {
                        "device_id": device_id,
                        "conversation_id": conversation_id
                    }}, "finish_reason": None}]}

        except Exception as e:
            logger.error(f"Error in pipe: {str(e)}")
            yield {"error": self._format_exception(e)}

    def _format_error(self, status_code: int, error: bytes) -> str:
        error_str = error.decode(errors="ignore") if isinstance(error, bytes) else error
        return json.dumps({"error": f"HTTP {status_code}: {error_str}"}, ensure_ascii=False)

    def _format_exception(self, e: Exception) -> str:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"}, ensure_ascii=False)

    def _nanoid(self, size=21) -> str:
        url_alphabet = "useandom-26T198340PX75pxJACKVERYMINDBUSHWOLF_GQZbfghjklqvwyzrict"
        random_bytes = secrets.token_bytes(size)
        return "".join([url_alphabet[b & 63] for b in reversed(random_bytes)])

    def _generate_sign(self, timestamp: str, payload: dict, nonce: str) -> str:
        payload_str = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        sign_str = f"{timestamp}{payload_str}{nonce}"
        return hashlib.md5(sign_str.encode("utf-8")).hexdigest().upper()

    async def _create_conversation(self, device_id: str) -> str:
        """创建新的会话"""
        payload = {"botCode": "AI_SEARCH"}
        timestamp = str(int(time.time()))
        nonce = self._nanoid(21)
        sign = self._generate_sign(timestamp, payload, nonce)

        headers = {
            "Origin": "https://ai.dangbei.com",
            "Referer": "https://ai.dangbei.com/",
            "User-Agent": self.user_agent,
            "deviceId": device_id,
            "nonce": nonce,
            "sign": sign,
            "timestamp": timestamp,
        }

        api = f"{config.API_DOMAIN}/ai-search/conversationApi/v1/create"
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(api, json=payload, headers=headers)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success"):
                        return data["data"]["conversationId"]
        except Exception as e:
            logger.error(f"Error creating conversation: {str(e)}")
        return None

# 创建实例
pipe = Pipe()

@app.post("/v1/chat/completions")
async def chat(request: ChatRequest, authorization: str = Header(None)):
    """
    OpenAI API 兼容的 Chat 端点
    """
    # 添加请求日志
    logger.info(f"Received chat request: {request.model_dump()}")
    
    await verify_api_key(authorization)

    # 使用 messages_as_dicts 属性
    request_data = request.model_dump()
    request_data['messages'] = request.messages_as_dicts

    async def response_generator():
        """流式响应生成器"""
        thinking_content = []
        is_thinking = False
        
        async for chunk in pipe.pipe(request_data):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    content = delta["content"]
                    if content == "<think>\n":
                        is_thinking = True
                    elif content == "\n</think>\n\n":
                        is_thinking = False
                    if is_thinking and content != "<think>\n":
                        thinking_content.append(content)
            
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    if request.stream:
        return StreamingResponse(response_generator(), media_type="text/event-stream")

    # 非流式响应
    content = ""
    meta = None
    try:
        async for chunk in pipe.pipe(request_data):
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    content += delta["content"]
                if "meta" in delta:
                    meta = delta["meta"]
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

    parts = content.split("\n\n\n", 1)
    reasoning_content = parts[0] if len(parts) > 0 else ""
    content = parts[1] if len(parts) > 1 else ""

    # 处理嵌套的 think 标签和特殊字符
    if reasoning_content:
        # 先尝试找到最外层的 think 标签
        start_idx = reasoning_content.find("<think>")
        end_idx = reasoning_content.rfind("</think>")
        
        if start_idx != -1 and end_idx != -1:
            # 如果找到了完整的外层标签，提取其中的内容
            inner_content = reasoning_content[start_idx + 7:end_idx].strip()
            # 移除内部的 think 标签
            inner_content = inner_content.replace("<think>", "").replace("</think>", "").strip()
            reasoning_content = f"<think>\n{inner_content}\n</think>"
        else:
            # 如果没有找到完整的标签，则移除所有 think 标签并重新添加
            reasoning_content = reasoning_content.replace("<think>", "").replace("</think>", "").strip()
            reasoning_content = f"<think>\n{reasoning_content}\n</think>"

    return {
        "id": str(uuid.uuid4()),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "message": {
                "role": "assistant",
                "reasoning_content": reasoning_content,
                "content": content,
                "meta": meta
            },
            "finish_reason": "stop"
        }]
    }

@app.get("/v1/models")
async def get_models(authorization: str = Header(None)):
    # 验证 API 密钥
    await verify_api_key(authorization)
    
    current_time = int(time.time())
    return {
        "object": "list",
        "data": [
            # 原始模型
            {
                "id": "DeepSeek-R1",
                "object": "model",
                "created": current_time,
                "owned_by": "library"
            },
            {
                "id": "DeepSeek-V3",
                "object": "model",
                "created": current_time,
                "owned_by": "library"
            },
            {
                "id": "Doubao",  # 改为大写开头
                "object": "model",
                "created": current_time,
                "owned_by": "library"
            },
            {
                "id": "Qwen",    # 改为大写开头
                "object": "model",
                "created": current_time,
                "owned_by": "library"
            },
            # 支持联网的模型
            {
                "id": "DeepSeek-R1-Search",
                "object": "model",
                "created": current_time,
                "owned_by": "library",
                "features": ["online_search"]
            },
            {
                "id": "DeepSeek-V3-Search",
                "object": "model",
                "created": current_time,
                "owned_by": "library",
                "features": ["online_search"]
            },
            {
                "id": "Doubao-Search",  # 改为大写开头
                "object": "model",
                "created": current_time,
                "owned_by": "library",
                "features": ["online_search"]
            },
            {
                "id": "Qwen-Search",    # 改为大写开头
                "object": "model",
                "created": current_time,
                "owned_by": "library",
                "features": ["online_search"]
            }
        ]
    }
@app.get("/")
def index():
    return "it's work!"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
