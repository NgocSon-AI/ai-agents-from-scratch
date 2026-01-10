def completions_create(client, messages: list, model: str) -> str:
    """
    Sends a request to client's `completions.create` method to interact with the language model

    Args:
        client (Groq): The Groq client object
        messages (list[dict]): A list of message objects containing chat history for the model.
        model (str): The model to use for generating tool calls and responses.
    
    Returns:
        str: The content of the model's response
    """

    """
    Hàm này sử dụng một instance của class Groq (client), gửi toàn bộ context đầu vào (ở đây là messages) cùng với tên mô hình (model) tới API của Groq bằng phương thức `chat.completions.create`, và nhận về response do model sinh ra dựa trên context đó.

    Args:
        client (Groq): Đối tượng client của class Groq.
        messages (list[dict]): Danh sách chứa các đối tượng `message` là lịch sử của hội thoại để cung cấp cho Model.
        model (str): Tên model được sử dụng cho tạo ra tool calls và responses.

    Returns:
        str: Nội dung response của model.
    """
    response = client.chat.completions.create(messages=messages, model=model)
    #print("Response: ", response)
    return str(response.choices[0].message.content)


def build_prompt_structure(prompt: str, role: str, tag: str = ""):
    """
    Builds a structured prompt that includes the role and content.

    Args:
        prompt(str): The actual content of the prompt.
        role (str): The role of the speaker (e.g., user, assistant).

    Returns:
        dict: A dictionary representing the structured prompt.
    """

    """
    Hàm này nhằm xây dựng môt prompt có cấu trúc bao gồm vai trò và nội dung.

    Args:
        prompt(str): Nội dung thực của prompt.
        role(str): Vai trò của người nói (ví dụ: người dùng và trợ lý)

    Returns:
        dict: Một dictionary biển diễn prompt có cấu trúc.    
    """
    if tag:
        prompt = f"<{tag}>{prompt}<{tag}>"
    return {"role": role, "content": prompt}

def update_chat_history(history: list, msg: str, role: str):
    """
    Updates the chat history by appending the latest response.

    Args:
        history (list): The list representing the curent chat history.
        msg (str): The message to append
        role (str): The role type (e.g. 'user', 'assistant', 'system')
    """

    """
    Hàm này để cập nhật lịch sử trò chuyện bằng cách thêm response mới vào.

    Args:
        history (list): Một cái danh sách biểu diễn hiện trạng của lịch sử chat
        msg (str): response được thêm vào.
        role (str): Loại vai trò (ví dụ: 'user', 'assistant', 'system') 
    """
    history.append(build_prompt_structure(prompt=msg, role=role))

class ChatHistory(list):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """
        Initialise the queue with a fixed total length.
        Khởi tạo Queue cùng với chiều dài tổng cố định
        
        Args:
            messages (list | None): A list of initial mesages
            messages (list | None): Một list các tin nhắn ban đầu, None nếu không có.
            total_length (int): The maximum number of message the chat history can 
            hold.
            total_length (int): Số lượng tin nhắn tối đa trong chat history có thể chứa.
        """
        if messages is None:
            messages = []

        super().__init__(messages)  # Gọi tới phương thức __init__() của lớp cha là list
        self.total_length = total_length

    def append(self, msg: str):
        """
        Add a message to the queue

        Args:
            msg (str): The message to be added to the queue.
        """

        """
        Thêm mt message vào queue

        Args:
            msg (str): Message được thêm vào queue
        """
        if len(self) == self.total_length:  # len(self) = self.__len__() = list.__len__(self) vì ChatHistory không có phương thức __len__ nên sẽ dùng len của list.
            self.pop(0)
        super().append(msg)
    
class FixedFirstChatHistory(ChatHistory):
    def __init__(self, messages: list | None = None, total_length: int = -1):
        """
        Initialise the queue with a fixed total length.

        Args:
            messages (list | None): A list of initial messages
            total_length (int): The maximum number of messages the chat history can hold.
        """

        """
        Khởi tạo queue cùng với kích thước cố định.

        Args:
            messages (list | None): Một list các tin nhắn ban đầu bằng None nếu không có tin nhắn.
            total_length (int): Số lượng tối đa các tin nhắn mà chat history có thể chứa.
        """
        super().__init__(messages, total_length)    # Gọi tới phương thức __init__() của ChatHistory

    def append(self, msg: str):
        """
        Add a message to the queue. The first message will always stay fixed.

        Args:
            msg (str): The message to be added to the queue
        """
        """
        Thêm tin nhắn vào queue. Tin nhắn đầu tin sẽ luôn luôn cố định.

        Args:
            msg (str): Tin nhắn được thêm vào queue.
        """
        if len(self) == self.total_length:      # len(self) = self.__len__() = list.__len__(self) vì ChatHistory không có phương thức __len__ nên sẽ dùng len của list.
            self.pop(1)
        super().append(msg) # Gọi tới phương thức append() của 