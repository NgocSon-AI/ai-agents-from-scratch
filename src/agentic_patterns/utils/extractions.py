import re
from dataclasses import dataclass

@dataclass  #là decorator nói với Python:
class TagContentResult:
    """
    A data class to represent the results of extracting tag content.

    Attributes:
        content (List(str)): A list of strings containing the content found between the specified tags.
        found (bool): A flag indicating whether any content was found for the given tag.
    """
    """
    Một data class để biểu diễn kết quả trích xuất nội dung tag
    Đây là một dataclass dùng để đóng gói kết quả trích xuất tag, với schema rõ ràng và ít boilerplate code.

    Attributes:
        content (List(str)): Danh sách các chuỗi chứa nội dung được tìm thấy giữa các thẻ được chỉ định.
        found (bool): Một cờ tín hiệu cho biết liệu có tìm thấy bất cứ nội dung nào ở giữa của thẻ không?
         
    """
    content: list[str]
    found: bool


def extract_tag_content(text: str, tag: str) -> TagContentResult:
    """
    Extracts all content enclosed by specified tags (e.g., <thought>, <response>, etc.).

    Args:
        text (str): The input string containing multiple potential tags.
        tag (str): The name of the tag to search for (e.g., 'thought', 'response')

    Returns:
        dict: A dictionary with the following keys:
            - 'content' (list): A list of strings containing the content found between the specificed tags.
            - 'found' (bool): A flag indicating whether any content was for the given tag.
    """

    """
    Trích xuất tất cả nội dung ở giữa các thẻ ví dụ như <thought></thought>, <response></response>

    Args:
        text (str): Chuỗi đầu vào chứa nhiều thẻ có tiềm năng.
        tag (str): Tên của các thẻ cần tìm.
    """
    # Build the regex pattern dynamically to find multiple occurrences of the tag
    # Xây dựng một biểu thức chính quy một cách linh hoạt để tìm ra nhiều lần xuất hiện của thẻ.
    tag_pattern = rf"<{tag}>(.*?)</{tag}>"

    # Use findall to capture all content between the specified tag
    # Sử dụng tìm tất cả để lấy tất cả nội dung ở giữa các thẻ cụ thể.
    matched_contents = re.findall(tag_pattern, text, re.DOTALL)

     # Return the dataclass instance with the result
    return TagContentResult(
        content = [content.strip() for content in matched_contents],
        found = bool(matched_contents),
    )