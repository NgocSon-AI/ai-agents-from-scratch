from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from src.agentic_patterns.utils.completions import completions_create
from src.agentic_patterns.utils.completions import build_prompt_structure
from src.agentic_patterns.utils.completions import FixedFirstChatHistory
from src.agentic_patterns.utils.completions import update_chat_history
from src.agentic_patterns.utils.logging import fancy_step_tracker

load_dotenv()

#Vai trò (role)
#Quy tắc an toàn
#Output discipline
#Định hình vai trò

#Giữ kỷ luật output

#Tránh hallucination

BASE_GENERATION_SYSTEM_PROMPT = """
Role: Bạn là một Chuyên gia Lập trình và Sáng tạo nội dung cấp cao.
Task: Thực hiện yêu cầu của người dùng với chất lượng cao nhất (State-of-the-art).

Quy trình:
1. Nếu là lần đầu: Hãy tạo nội dung giải quyết triệt để yêu cầu, đảm bảo tính chính xác, hiệu suất và dễ hiểu.
2. Nếu nhận được bản Critique (Phê bình): 
   - Phân tích kỹ từng điểm phê bình.
   - Trả về bản nội dung ĐÃ CẢI THIỆN (Revised Version) tích hợp toàn bộ các kiến nghị.
   - Chỉ ra ngắn gọn những điểm quan trọng bạn đã nâng cấp ở đầu phản hồi.

Yêu cầu định dạng: Sử dụng Markdown rõ ràng, mã nguồn phải có chú thích (comment) và hướng dẫn sử dụng chi tiết.
"""
BASE_REFLECTION_SYSTEM_PROMPT = """
Role: Bạn là một Giám khảo Kỹ thuật khắt khe và là Người tối ưu hóa quy trình.
Task: Đánh giá nội dung được cung cấp dựa trên các tiêu chí: Tính chính xác, Hiệu suất, Khả năng mở rộng và Trải nghiệm người dùng.

Quy trình đánh giá:
1. Tìm lỗi logic hoặc lỗi cú pháp (nếu có).
2. Tìm các điểm có thể tối ưu (Refactor): Ví dụ như giảm độ phức tạp thuật toán, cải thiện khả năng đọc của code, hoặc làm sâu sắc thêm nội dung giải thích.
3. Kiểm tra tính an toàn: Các lỗ hổng bảo mật hoặc rủi ro về bộ nhớ/dữ liệu.

Định dạng đầu ra:
- Nếu cần cải thiện: Liệt kê danh sách các [Critique] (Điểm yếu hiện tại) và [Recommendation] (Cách khắc phục cụ thể). Hãy cụ thể, không góp ý chung chung.
- Nếu nội dung đã thực sự hoàn hảo và không thể tối ưu thêm: Chỉ trả về duy nhất từ khóa: <OK>.
"""

class ReflectionAgent:
    
    """
    Class ReflectionAgent triển khai một Agent nhằm tạo ra các responses và tự reflection về chính những response đó. Bằng cách sử dụng LLM để cải thiện nhiều lần.

    Đầu tiên thì Agent sẽ tạo ra response dựa trên prompt được cung cấp, sau đó sẽ phê bình, nhận xét, đánh giá chúng trong bước Reflection.

    Attributes:
        model (str): Tên của model được sử dụng để tạo ra response và tự phản tư.
        client (str): Đối thực thể của client Groq() để tương tác với các Language Model.
    """
    def __init__(self, model:str = "llama-3.1-8b-instant"):
        self.client = Groq()
        self.model = model

    def _request_completion(
            self,
            history: list,
            verbose: int = 0,
            log_title: str = "****************COMPLETION****************",
            log_color: str = ""
    ) -> str:
        """
        Một phương thưc private để request Groq model tạo ra một completion

        Args:
            + history (list): Danh sách chứa các message lịch sử chat với model.
            + verbose (int, optional): Mức độ chi tiết của log, dùng để kiểm soát việc in thông tin ra màn hình. Mặc định là 0.
            + log_title (str, optional): Tiêu đề của log. Default là "COMPLETION".
            + log_color (str, optional): Màu sắc hiển thị log. Mặc định là None "".

        Returns:
            + str: Response do model sinh ra. 
        """

        output = completions_create(client=self.client, messages=history, model=self.model)

        #print("OUPUT: ", output)

        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)
        
        return output
    

    def generation(
            self,
            generation_history: list,
            verbose: int = 0
    ) -> str:
        """
        Tạo ra một response dựa trên `generation history` đã được cung cấp bằng cách sử dụng Language Model.

        Args:
            + generation_history (list): Danh sách các message tạo thành lịch sử hội thoại hoặc lịch sử tạo nội dung.
            + verbose (int, optional): Mức độ chi tiết của log, dùng để kiểm soát thông tin in ra màn hình. Mặc định là 0.

        Returns:
            str: Response đã được tạo ra.
        """
        return self._request_completion(
            history=generation_history,
            verbose=verbose,
            log_title="****************GENERATION****************",
            log_color=Fore.CYAN,
        )
    

    def reflection(
            self,
            reflection_history: list,
            verbose: int=0
    ) -> str:
        """
        Thực hiện reflection trên content generation history bằng cách tạo ra một bản nhận xét, đánh giá response.

        Args:
            + reflection_history (list): Danh sách các message tạo thành reflection history, dựa trên kết quả content generation hoặc tương tác trước đó.
            + verbose (int, optional): Mức độ chi tiết của log, dùng để kiểm soát thông tin in ra màn hình. Mặc định là 0.

        Returns:
            + str: Reflection response do model tạo ra.
        """
        return self._request_completion(
            history=reflection_history,
            verbose=verbose,
            log_title= "****************REFLECTION****************",
            log_color=Fore.GREEN
        )
    

    def run(
            self,
            user_message: str,
            generation_system_prompt: str="",
            reflection_system_prompt: str="",
            n_steps: int=10,
            verbose: int=0
    ) -> str:
        """
        Chạy ReflectionAgent trong nhiều bước, luân phiên giữa việc tạo phản hồi và phản chiếu (đánh giá) phản hồi đó trong số bước được chỉ định.

        Args:
            + user_message (str): Tin nhắn hoặc truy vấn của người dùng
            + generation_system_prompt (str, optional): Prompt hệ thống dùng để hướng dẫn quá trình tạo văn bản. 
            + reflection_system_prompt (str, optional): Prompt hệ thống dùng để hướng dẫn quá trình phản tư/đánh giá.
            + n_steps (int, optional): Số vòng luân phiên của quá trình Agent tạo và phản tư. 
            + verbose (int, optional): Mức độ chi tiết của log, thông tin in ra màn hình. Mặc định là 0.
        
        Returns:
            + str: Response cuối cùng được tạo ra sau tất cả các vòng lặp.
        """
        # Nhằm kết hợp prompt base và prompt từ người dùng. Nếu user không truyền thì mặc định sử dụng prompt base
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT   
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_message, role="user")
            ],
            total_length=3
        )

        reflection_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=reflection_system_prompt, role="system")
            ],
            total_length=3
        )
        
        for step in range(n_steps):
            if verbose > 0:
                # Theo dõi vòng lặp
                fancy_step_tracker(step=step, total_steps=n_steps)

            generation = self.generation(generation_history=generation_history, verbose=verbose)

            update_chat_history(history=generation_history, msg=generation, role="assistant")
            update_chat_history(history=reflection_history, msg=generation, role="user")

            critique = self.reflection(reflection_history=reflection_history, verbose=verbose)

            if "<OK>" in critique:
                # Nếu không có sư thay đổi, bổ sung, dừng loop
                print(
                    Fore.MAGENTA,
                    "\n\nĐã tìm thấy trình tự dừng. Dừng vòng lắp ... \n\n",
                )
                break

            update_chat_history(history=generation_history, msg=critique, role="user")
            update_chat_history(history=reflection_history, msg=critique, role="assistant")

        return generation


        
  