import asyncio
import pathlib

import telegramify_markdown
from telegramify_markdown.interpreters import (
    InterpreterChain,
    TextInterpreter,
)
from telegramify_markdown.type import ContentTypes

md = pathlib.Path(__file__).parent.joinpath("t_longtext.md").read_text(encoding="utf-8")


async def send_message():
    interpreter_chain = InterpreterChain([TextInterpreter()])

    boxs = await telegramify_markdown.telegramify(
        content=md,
        interpreters_use=interpreter_chain,
        latex_escape=True,
        normalize_whitespace=True,
        max_word_count=4090,
    )
    for item in boxs:
        if item.content_type == ContentTypes.TEXT:
            print(item.content)


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    loop.run_until_complete(send_message())
