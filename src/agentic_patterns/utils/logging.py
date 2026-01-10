import time
from colorama import Fore
from colorama import Style


def fancy_print(message: str) -> None:
    """
    Displays a fancy print message.

    Args:
        message (str): The message to display.
    """

    """
    Hiển thị màu sắc cho message

    Args:
        message (str): Message được hiển 
    """
    print(Style.BRIGHT + Fore.CYAN + f"\n{'=' * 50}")
    print(Fore.MAGENTA + f"{message}")
    print(Style.BRIGHT + Fore.CYAN + f"{'=' * 50}\n")
    time.sleep(0.5)



def fancy_step_tracker(step: int, total_steps: int) -> None:
    """
    Displays a fancy step tracker for each iteration of the generation-reflection loop.

    Args:
        step (int): The current step in the loop.
        total_steps (int): The total number of steps in the loop.
    """

    """
    Trình theo dõi các bước cho mỗi lần lặp của vòng lặp phản tư

    Args:
        step (int): Bước lặp hiện tại
        total_steps (int): Tổng số bước của vòng lặp
    """
    fancy_print(f"STEP {step + 1}/{total_steps}")

