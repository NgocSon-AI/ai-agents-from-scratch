from colorama import Fore
from dotenv import load_dotenv
from groq import Groq

from src.agentic_patterns.utils.completions import completions_create
from src.agentic_patterns.utils.completions import build_prompt_structure
from src.agentic_patterns.utils.completions import FixedFirstChatHistory
from src.agentic_patterns.utils.completions import update_chat_history
from src.agentic_patterns.utils.logging import fancy_step_tracker

load_dotenv()

BASE_GENERATION_SYSTEM_PROMPT = """
Nhiệm vụ của bạn là tạo ra nội dung tốt nhất có thể đáp ứng yêu cầu của người dùng.
Nếu người dùng đưa ra phản hồi, hãy gửi lại phiên bản đã chỉnh sửa của bài làm trước đó.
Bạn phải luôn luôn xuất ra nội dung đã được chỉnh sửa.
"""
BASE_REFLECTION_SYSTEM_PROMPT = """
Bạn được giao nhiệm vụ đưa ra nhận xét và đề xuất về nội dung do người dùng tạo ra.
Nếu nội dung của người dùng có lỗi hoặc cần cải thiện, hãy xuất ra danh sách các đề xuất
và nhận xét. Nếu nội dung của người dùng ổn và không có gì cần thay đổi, hãy xuất ra: <OK>
"""

class ReflectionAgent:
    """A class that implements a Reflection Agent, which generates responses and reflects
    on them using the LLM to iteratively improve the interaction. The agent first generates
    responses based on provided prompts and then critiques them in a reflection step.
    Attributes:
        model (str): The model name used for generating and reflecting on responses.
        client (Groq): An instance of the Groq client to interact with the language model.
    """
    def __init__(self, model: str="llama-3.1-8b-instant"):
        self.client = Groq()
        self.model = model

    def _request_completion(
            self,
            history: list,
            verbose: int = 0,
            log_title: str = "COMPLETION",
            log_color: str = "",        
    ) -> str:
        """
        A private method to request a completion from the Groq model.

        Args:
            history (list): A list of message forming the conversation or reflection history
            verbose (int, optional): The verbosity level. Defaults to 0 (no output).
            log_title (str, optional): _description_. Defaults to "COMPLETION".
            log_color (str, optional): _description_. Defaults to "".
        Returns:
            str: The model-generated response.
        """
        output = completions_create(self.client, history, self.model)

        if verbose > 0:
            print(log_color, f"\n\n{log_title}\n\n", output)
        
        return output
    
    def generate(self, generation_history: list, verbose: int = 0) -> str:
        """Generates a response based on the provided generation history using the model.


        Args:
            generation_history (list): A list of messages forming the conversation or generation history.
            verbose (int, optional): The verbosity level, controlling printed output. Defaults to 0.

        Returns:
            str: _description_
        """
        return self._request_completion(
            generation_history, verbose, log_title="GENERATION", log_color=Fore.CYAN
        )
    
    def reflect(self, reflection_history: list, verbose: int=0) -> str:
        """Reflects on the generation history by generating a critique or feedback.

        Args:
            reflection_history (list): A list of messages forming the reflection history, typically based on the previous generation or interaction.
            verbose (int, optional): The verbosity level, controlling printed output. Defaults to 0.

        Returns:
            str: The critique or reflection response from the model.
        """
        return self._request_completion(
            reflection_history, verbose, log_title="REFLECTION", log_color=Fore.GREEN
        )
    
    def run(
            self,
            user_msg: str,
            generation_system_prompt: str="",
            reflection_system_prompt: str="",
            n_steps: int=10,
            verbose: int=0,
    ) -> str:
        """Runs the ReflectionAgent over multiple steps, alternating between generating a response and reflecting on it for the specified number of steps.

        Args:
            user_msg (str): The user message or query that initiates the interaction.
            generation_system_prompt (str, optional): The system prompt for guiding the generation process.
            reflection_system_prompt The system prompt for guiding the reflection process.
            n_steps (int, optional): The number of generate-reflect cycles to perform. Defaults to 3.
            verbose (int, optional): The verbosity level controlling printed output. Defaults to 0.

        Returns:
            str: The final generated response after all cycles are completed
        """
        generation_system_prompt += BASE_GENERATION_SYSTEM_PROMPT
        reflection_system_prompt += BASE_REFLECTION_SYSTEM_PROMPT

        generation_history = FixedFirstChatHistory(
            [
                build_prompt_structure(prompt=generation_system_prompt, role="system"),
                build_prompt_structure(prompt=user_msg, role="user")
            ],
            total_length=3,
        )

        reflection_history = FixedFirstChatHistory(
            [build_prompt_structure(prompt=reflection_system_prompt, role="system")],
            total_length=3,
        )

        for step in range(n_steps):
            if verbose > 0:
                fancy_step_tracker(step, n_steps)

                generation = self.generate(generation_history=generation_history, verbose=verbose)
                update_chat_history(generation_history, generation, "assistant")
                update_chat_history(reflection_history, generation, "user")

                critique = self.reflect(reflection_history, verbose=verbose)

                if "<OK>" in critique:
                    print(
                        Fore.RED,
                        "\n\nStop Sequence fond. Stoping the reflection loop...\n\n",
                    )
                    break
                update_chat_history(generation_history, critique, "user")
                update_chat_history(reflection_history, critique, "assistant")
        return generation