import asyncio
import sys

import telegramify_markdown
from telegramify_markdown.interpreters import InterpreterChain, TextInterpreter
from telegramify_markdown.type import ContentTypes


async def main():
    # Read entire stdin as the markdown source.
    md = sys.stdin.read()

    interpreter_chain = InterpreterChain([TextInterpreter()])

    boxes = await telegramify_markdown.telegramify(
        content=md,
        interpreters_use=interpreter_chain,
        latex_escape=True,
        normalize_whitespace=True,
        max_word_count=4090,
    )

    for item in boxes:
        if item.content_type == ContentTypes.TEXT:
            # Write each text block as UTF-8 bytes, followed by a NUL separator.
            sys.stdout.buffer.write(item.content.encode("utf-8"))
            sys.stdout.buffer.write(b"\0")

    sys.stdout.flush()


if __name__ == "__main__":
    asyncio.run(main())
