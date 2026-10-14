"""Minh hoạ FUNCTION CALLING thuần.

Tool `get_weather` được định nghĩa schema thủ công VÀ thực thi ngay trong
chính file app này. Model chỉ QUYẾT ĐỊNH gọi tool nào; app mới là nơi chạy.

Hỗ trợ cả:
1. Cấu hình .env (OpenAI compatible qua 9router / proxy)
2. Google Gemini SDK thuần (nếu có GEMINI_API_KEY)
"""

import json
import os
import sys
from pathlib import Path

# Cấu hình encoding UTF-8 cho stdout trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Load file .env từ thư mục gốc hoặc thư mục hiện tại
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

SYSTEM_INSTRUCTION = (
    "Bạn là trợ lý thời tiết thân thiện, trả lời bằng tiếng Việt tự nhiên. "
    "Dùng emoji phù hợp (🌧️ 🌤️ 💨 💧). "
    "Tóm tắt ngắn gọn, dễ hiểu, và đưa ra lời khuyên thực tế "
    "(ví dụ: mang ô, mặc áo mỏng, ...)."
)

# 2. App tự thực thi tool (trong thực tế sẽ gọi API thời tiết thật)
def get_weather(city: str) -> str:
    """Trả về thời tiết (mock) của *city*. Dùng làm tool cho model."""
    mock_data = {
        "Hà Nội": {
            "nhiệt_độ": "29°C",
            "thời_tiết": "trời mưa nhẹ",
            "độ_ẩm": "82%",
            "gió": {"hướng": "Đông Nam", "tốc_độ": "12 km/h"},
        },
        "Hồ Chí Minh": {
            "nhiệt_độ": "33°C",
            "thời_tiết": "mưa rào",
            "độ_ẩm": "75%",
            "gió": {"hướng": "Tây Nam", "tốc_độ": "15 km/h"},
        },
        "Đà Nẵng": {
            "nhiệt_độ": "30°C",
            "thời_tiết": "nhiều mây",
            "độ_ẩm": "78%",
            "gió": {"hướng": "Đông", "tốc_độ": "10 km/h"},
        },
    }
    default = {"nhiệt_độ": "28°C", "thời_tiết": "không có dữ liệu chi tiết"}
    return json.dumps({"city": city, **mock_data.get(city, default)}, ensure_ascii=False)


def run_openai(prompt: str) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    base_url = os.getenv("OPENAI_BASE_URL")
    model = os.getenv("MODEL", "gpt-4o-mini")

    client = OpenAI(api_key=api_key, base_url=base_url)

    # 1. App tự định nghĩa schema của tool (OpenAI JSON Schema)
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Lấy thời tiết hiện tại của một thành phố",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Tên thành phố (ví dụ: Hà Nội, Đà Nẵng, Hồ Chí Minh)"
                    }
                },
                "required": ["city"],
            },
        }
    }]

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": prompt}
    ]

    # 3. Gọi model — model quyết định có gọi tool hay không
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        tools=tools,
    )

    msg = resp.choices[0].message

    # 4. Vòng lặp: nếu model yêu cầu tool, app TỰ THỰC THI rồi đưa kết quả trả lại
    while msg.tool_calls:
        messages.append(msg)
        for tc in msg.tool_calls:
            fc_name = tc.function.name
            fc_args = json.loads(tc.function.arguments)
            print(f"  [model yêu cầu] {fc_name}({fc_args})")
            
            # App chạy tool thật
            result = get_weather(**fc_args)
            print(f"  [app thực thi]  -> {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "name": fc_name,
                "content": result
            })

        # Gửi lại kết quả tool cho model
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
        )
        msg = resp.choices[0].message

    # 5. Model tổng hợp câu trả lời cuối
    return msg.content or ""


def run_gemini(prompt: str) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    model = "gemini-2.5-flash"

    # 1. App tự định nghĩa schema của tool
    get_weather_declaration = types.FunctionDeclaration(
        name="get_weather",
        description="Lấy thời tiết hiện tại của một thành phố",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "city": types.Schema(
                    type=types.Type.STRING, description="Tên thành phố"
                )
            },
            required=["city"],
        ),
    )
    tools = [types.Tool(function_declarations=[get_weather_declaration])]

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    ]

    # 3. Gọi model
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            tools=tools,
            system_instruction=SYSTEM_INSTRUCTION,
        ),
    )

    # 4. Vòng lặp
    while resp.function_calls:
        contents.append(resp.candidates[0].content)
        function_responses = []
        for fc in resp.function_calls:
            print(f"  [model yêu cầu] {fc.name}({fc.args})")
            result = get_weather(**fc.args)
            print(f"  [app thực thi]  -> {result}")
            function_responses.append(
                types.Part.from_function_response(
                    name=fc.name, response={"result": result}
                )
            )

        contents.append(types.Content(role="user", parts=function_responses))
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                tools=tools,
                system_instruction=SYSTEM_INSTRUCTION,
            ),
        )

    # 5. Model tổng hợp câu trả lời cuối
    return resp.text or ""


def run(prompt: str) -> str:
    """Tự động chọn client dựa trên cấu hình môi trường (.env hoặc GEMINI_API_KEY)."""
    if os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_BASE_URL"):
        return run_openai(prompt)
    return run_gemini(prompt)


if __name__ == "__main__":
    question = "Thời tiết Hà Nội và Đà Nẵng hôm nay thế nào?"
    print(f"User: {question}\n")
    print("Trả lời:", run(question))
