import json
from typing import Callable


def get_fn_signature(fn: Callable) -> dict:
    """
    Tạo chữ ký cho một hàm cho trước.


    Args:
        fn (Callable): Hàm cần trích xuất chữ ký.

    Returns:
        dict: Một từ điển chứa tên hàm, mô tả, và kiểu tham số.
    """
    fn_signature: dict = {
        "name": fn.__name__,    
        "description": fn.__doc__,
        "parameters": {
            "properties": {
                #"a" : {"type": "int"},
                #"s" : {"type": "str"}
            }
        },
    }
    schema = {
        k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"
    }
    fn_signature["parameters"]["properties"] = schema   # Truyen schema vao properties cua fn_signature

    return fn_signature


def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """
    Validates and converts arguments in the input dictionary to match the expected types.
    Sửa dữ liệu AI gửi vào cho đúng kiểu

    Args:
        tool_call (dict): A dictionary containing the arguments passed to the tool. 
        Một từ điển chứa các tham số được truyền cho công cụ.
        tool_signature (dict): The expected function signature and parameter types.
        Chữ ký hàm dự kiến và kiểu tham số.

    Returns:
        dict: The tool call dictionary with the arguments converted to the correct types if necessary.
        Công cụ này gọi từ điển với các đối số được chuyển đổi sang kiểu dữ liệu chính xác nếu cần.
    """
    properties = tool_signature["parameters"]["properties"]

    # TODO: This is overly simplified but enough for simple Tools.
    type_mapping = {
        "int": int,
        "str": str,
        "bool": bool,
        "float": float,
    }

    for arg_name, arg_value in tool_call["arguments"].items():
        expected_type = properties[arg_name].get("type")

        if not isinstance(arg_value, type_mapping[expected_type]):
            tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)

    return tool_call


class Tool:
    """
    A class representing a tool that wraps a callable and its signature.
    Một lớp đại diện cho một công cụ bao bọc một hàm có thể gọi được và chữ ký của nó.

    Attributes:
        name (str): The name of the tool (function).
        fn (Callable): The function that the tool represents. Chức năng mà công cụ đó thể hiện.
        fn_signature (str): JSON string representation of the function's signature. Chuỗi JSON biểu diễn chữ ký của hàm.
    """

    def __init__(self, name: str, fn: Callable, fn_signature: str):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature

    def __str__(self):
        return self.fn_signature

    def run(self, **kwargs):
        """
        Executes the tool (function) with provided arguments.

        Args:
            **kwargs: Keyword arguments passed to the function.

        Returns:
            The result of the function call.
        """
        return self.fn(**kwargs)


def tool(fn: Callable):
    """
    A decorator that wraps a function into a Tool object.
    Giúp bạn đỡ phải viết nhiều code
    Args:
        fn (Callable): The function to be wrapped.

    Returns:
        Tool: A Tool object containing the function, its name, and its signature.
    """

    def wrapper():
        fn_signature = get_fn_signature(fn)
        return Tool(
            name=str(fn_signature.get("name")),
            fn=fn,
            fn_signature=json.dumps(fn_signature)
        )

    return wrapper()