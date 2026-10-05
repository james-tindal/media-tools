"""Convert CommonMark Markdown into the text passed to speech synthesis."""

from markdown_it import MarkdownIt

CONVERSION = "markdown-it-commonmark-inline-v1"
MD = MarkdownIt("commonmark")


def markdown_to_text(source):
    paragraphs = []
    for token in MD.parse(source):
        if token.type != "inline":
            continue
        text = "".join(
            child.content if child.type in {"text", "code_inline", "image"}
            else "\n" if child.type in {"softbreak", "hardbreak"}
            else ""
            for child in token.children or []
        )
        if text.strip():
            paragraphs.append(text.strip())
    return "\n\n".join(paragraphs)
