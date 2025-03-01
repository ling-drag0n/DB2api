# Dangbei AI Chat API

基于 FastAPI 的 AI 聊天服务，提供 OpenAI 兼容的 API 接口。

## 主要功能

- 支持多种 AI 模型：
  - DeepSeek-R1/V3
  - Doubao
  - Qwen
  - 以上模型均支持联网搜索版本（如 DeepSeek-R1-Search）
- 支持流式输出
- 支持多轮对话
- 支持联网搜索

## 快速开始

### Docker 部署

```bash
# 启动服务
docker-compose up -d

# 查看日志
docker-compose logs -f

# 停止服务
docker-compose down
```

服务将在 http://localhost:8000 运行

### API 使用示例

#### 1. 获取可用模型

```bash
curl http://localhost:8000/v1/models \
  -H "Authorization: Bearer sk_gUXNcLwm0rnnEt55Mg8hq88"
```

#### 2. 基础对话

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk_gUXNcLwm0rnnEt55Mg8hq88" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3",
    "messages": [
      {"role": "user", "content": "你好"}
    ]
  }'
```

#### 3. 联网搜索

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk_gUXNcLwm0rnnEt55Mg8hq88" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3-Search",
    "messages": [
      {"role": "user", "content": "最近的新闻有哪些？"}
    ]
  }'
```

#### 4. 流式输出

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer sk_gUXNcLwm0rnnEt55Mg8hq88" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-V3",
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "stream": true
  }'
```

## 配置说明

在 docker-compose.yml 中配置环境变量：

```yaml
environment:
  - API_KEY=sk_gUXNcLwm0rnnEt55Mg8hq88  # API 认证密钥
  - API_DOMAIN=https://ai-api.dangbei.net # API 域名
```

## 注意事项

- 生产环境请修改默认的 API Key
- 确保服务器有足够的资源运行服务
- 联网搜索功能仅支持带 "-Search" 后缀的模型
