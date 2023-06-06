from typing import List, Optional

from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
import requests
import json

class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.1
    top_p = 0.9
    history = []

    def __init__(self):
        super().__init__()

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # headers中添加上content-type这个参数，指定为json格式
        headers = {'Content-Type': 'application/json'}
        data = {
            'prompt': prompt,
            'temperature': self.temperature,
            'history': self.history,
            'max_length': self.max_token
        }
        # 调用api
        response = requests.post("http://127.0.0.1:8000", headers=headers, json=data)
        if response.status_code != 200:
            return "查询结果错误"
        resp = response.json()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)
        self.history = self.history + [[None, resp['response']]]
        return resp['response']


if __name__ == '__main__':
    import requests

    D = {"prompt": "", "history": []}
    D["prompt"] = " 感冒发烧怎么引起的?"
    print(D)
    data = requests.post("http://127.0.0.1:8000", json=D, headers={"Content-Type": "application/json"})
    print(data.text)
