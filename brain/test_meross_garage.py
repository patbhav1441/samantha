import os
from dotenv import load_dotenv

from src import tools


def main():
    load_dotenv()
    print("Testing LEFT garage OPEN...")
    res_open = tools.tool_garage_open(door="left")
    print("OPEN result:", res_open)

    input("Press Enter to CLOSE LEFT...")
    res_close = tools.tool_garage_close(door="left")
    print("CLOSE result:", res_close)

    print("Testing RIGHT garage OPEN...")
    res_open = tools.tool_garage_open(door="right")
    print("OPEN result:", res_open)

    input("Press Enter to CLOSE RIGHT...")
    res_close = tools.tool_garage_close(door="right")
    print("CLOSE result:", res_close)


if __name__ == "__main__":
    main()

