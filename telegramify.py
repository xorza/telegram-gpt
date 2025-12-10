import asyncio
import sys

import telegramify_markdown
from telegramify_markdown.interpreters import InterpreterChain, TextInterpreter
from telegramify_markdown.type import ContentTypes


def main() -> int:
    md = sys.stdin.read()

    interpreter_chain = InterpreterChain([TextInterpreter()])

    async def run():
        return await telegramify_markdown.telegramify(
            content=md,
            interpreters_use=interpreter_chain,
            latex_escape=True,
            normalize_whitespace=True,
            max_word_count=4090,
        )

    # Block until the async processing finishes, but keep a simple sync interface.
    boxes = asyncio.run(run())

    for item in boxes:
        if item.content_type == ContentTypes.TEXT:
            # Write each text block as UTF-8 bytes, followed by a NUL separator.
            sys.stdout.buffer.write(item.content.encode("utf-8"))
            sys.stdout.buffer.write(b"\0")

    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    main()
