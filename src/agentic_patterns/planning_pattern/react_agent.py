import json
import re

from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from agentic_patterns.tool_pattern.tool import Tool
from agentic_patterns.tool_pattern.tool import validate_arguments
from agentic_patterns.utils.completions import build_prompt_structure
from agentic_patterns.utils.completions import ChatHistory
from agentic_patterns.utils.completions import completions_create
from agentic_patterns.utils.completions import update_chat_history
from agentic_patterns.utils.extractions import extract_tag_content


load_dotenv()


BASE_SYSTEM_PROMPT = ""

REACT_SYSTEM_PROMPT = """
Bạn vận hành bằng cách chạy một vòng lặp với các bước sau: Suy nghĩ, Hành động, Quan sát.
Bạn được cung cấp các chữ ký hàm trong thẻ XML <tools></tools>.
Bạn có thể gọi một hoặc nhiều hàm để hỗ trợ truy vấn của người dùng. Đừng đưa ra giả định về giá trị cần truyền vào hàm.
Hãy đặc biệt chú ý đến thuộc tính "types". Bạn nên sử dụng các kiểu dữ liệu đó giống như trong một từ điển Python.
Với mỗi lần gọi hàm, hãy trả về một đối tượng JSON chứa tên hàm và các tham số bên trong thẻ XML <tool_call></tool_call> như sau:
<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>
Dưới đây là các công cụ/thao tác có sẵn:
<tools>
%s
</tools>
Ví dụ về phiên làm việc:
<question>Nhiệt độ hiện tại ở Hanoi là bao nhiêu?</question>
<thought>Tôi cần lấy thông tin thời tiết hiện tại ở Hanoi</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Hanoi", "unit": "C"}, "id": 0}</tool_call>
Bạn sẽ được gọi lại với thông tin này:
<observation>{0: {"temperature": 25, "unit": "C"}}</observation>
Sau đó, bạn xuất ra:
<response>Nhiệt độ hiện tại ở Hanoi là 25 độ C</response>
Các ràng buộc bổ sung:
- Nếu người dùng hỏi bạn điều gì đó không liên quan đến bất kỳ công cụ nào ở trên, hãy trả lời tự do bằng cách đặt câu trả lời của bạn trong thẻ <response></response>.
"""


class ReactAgent:
    """
    Một lớp đại diện cho một tác nhân sử dụng logic ReAct để tương tác với các công cụ nhằm xử lý
    đầu vào của người dùng, đưa ra quyết định và thực thi các lệnh gọi công cụ. Tác nhân có thể chạy các phiên tương tác,
    thu thập chữ ký công cụ và xử lý nhiều lệnh gọi công cụ trong một vòng tương tác nhất định.

    Attributes:
        client (Groq): Ứng dụng khách Groq trước đây xử lý việc hoàn thành dựa trên mô hình.
        model (str): Tên của mô hình được sử dụng để tạo ra các phản hồi. Mặc định là "llama-3.3-70b-versatile".
        tools (list[Tool]): Danh sách các phiên bản công cụ có sẵn để thực thi.
        tools_dict (dict): Một từ điển ánh xạ tên công cụ với các thể hiện công cụ tương ứng.
    """

    def __init__(self, tools: Tool | list[Tool], model: str = "llama-3.3-70b-versatile", system_prompt: str="") -> None:
        """
        Khởi tạo ReactAgent với các công cụ và mô hình được cung cấp.

        Args:
            tools (Tool | list[Tool]): Một thể hiện duy nhất của Công cụ hoặc một danh sách các thể hiện của Công cụ
            model (str): Tên của mô hình sẽ được sử dụng để tạo ra các phản hồi.
        """
        self.client = Groq()
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """Thu thập chữ ký hàm của tất cả các công cụ có sẵn.

        Returns:
            str: Một chuỗi kết hợp tất cả các chữ ký công cụ ở định dạng JSON.
        """
        return "".join([tool.fn_signature for tool in self.tools])
    
    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """Chương trình xử lý từng lệnh gọi công cụ, xác thực các tham số, thực thi các công cụ và thu thập kết quả.

        Args:
            tool_calls_content (list): Danh sách các chuỗi ký tự, mỗi chuỗi đại diện cho một lệnh gọi công cụ ở định dạng JSON.

        Returns:
            dict: Một từ điển trong đó khóa là ID lệnh gọi công cụ và giá trị là kết quả từ các công cụ đó.
        """
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")

            # Validate and execute the tool call
            validated_tool_call = validate_arguments(
                tool_call, json.loads(tool.fn_signature)
            )
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")

            observations[validated_tool_call["id"]] = result

        return observations
    

    def run(self, user_msg: str, max_rounds: int = 10) -> str:
        """
        Thực hiện một phiên tương tác với người dùng, trong đó agent xử lý đầu vào của người dùng, tạo phản hồi,
        xử lý các cuộc gọi tool và cập nhật lịch sử trò chuyện cho đến khi có phản hồi cuối cùng hoặc đạt đến số vòng tối đa.

        Args:
            user_msg (str): Thông điệp do người dùng nhập vào để bắt đầu tương tác.
            max_rounds (int, optional): Số vòng tương tác tối đa mà Agent nên thực hiện. Mặc định là 10.

        Returns:
            str: Phản hồi cuối cùng được tạo ra bởi agent sau khi xử lý dữ liệu đầu vào của người dùng và bất kỳ lệnh gọi tool nào.
        """
        user_prompt = build_prompt_structure(
            prompt=user_msg,
            role="user",
            tag="question"
        )
        if self.tools:
            self.system_prompt += (
                "\n" + REACT_SYSTEM_PROMPT % self.add_tool_signatures()
            )

        chat_history = ChatHistory(
            [
                build_prompt_structure(
                    prompt=self.system_prompt,
                    role="system"
                ),
                user_prompt
            ]
        )

        if self.tools:
            for _ in range(max_rounds):
                completion = completions_create(self.client, chat_history, self.model)

                response = extract_tag_content(str(completion), "response")
                if response.found:
                    return response.content[0]
                
                thought = extract_tag_content(str(completion), "thought")
                tool_calls = extract_tag_content(str(completion), "tool_call")

                update_chat_history(chat_history, completion, role="assistant")

                print(Fore.MAGENTA + f"\nAgent Thought: \n{thought.content[0]}")

                if tool_calls.found:
                    observations = self.process_tool_calls(tool_calls.content)

                    print(Fore.BLUE + f"\nObservations: \n{observations}")
                    
                    update_chat_history(chat_history, f"{observations}", "user")

        return completions_create(self.client, chat_history, self.model)

