from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import os

# 加载 .env 文件
load_dotenv(override=True)

def message_handle(messages, retain_num = 3):
    """
        如果消息列表大于 retain_num * 2，则将历史对话消息删除
    """

    if len(messages) > retain_num * 2:
        del messages[1:len(messages) - retain_num * 2]

def chatbox_main():
    """
        聊天机器人的入口方法
    """

    # 初始化 DeepSeek 模型
    model = init_chat_model(
        model = "deepseek:deepseek-v4-flash",
        api_key = os.getenv("DEEPSEEK_API_KEY")
    )

    # 初始化消息列表
    messages = [
        {"role":"system", "content":os.getenv("SYSTEM_PROMPT")}
    ]

    idx = 1
    while True:
        user_message = input(f"[第{idx}轮对话]请输入(exit 退出)：")

        if user_message == 'exit':
            print("退出对话".center(50, "="))
            break

        # 将用户消息追加到消息列表
        messages.append({"role":"user", "content":user_message})

        # 调用大模型回答
        response = model.stream(messages)

        response_content = ""
        print(f"[第{idx}轮对话]回答：", end="")
        for item in response:
            if item.content:
                print(item.content, end="", flush=True)
                response_content += item.content

        print("\n") # 换行

        # 将回答内容追加到消息列表(添加记忆)
        messages.append({"role":"assistant", "content":response_content})

        # 将过长的消息列表优化截取
        message_handle(messages)

        idx += 1 # 自增

if __name__ == '__main__':
    chatbox_main()