from arclet.entari.logger import log as log_m

logger = log_m.wrapper("[llm]")


def log(level: str, rich_text: str) -> None:
    getattr(logger.opt(colors=True), level)(
        rich_text.replace("[", "<").replace("]", ">"),
        alt=rich_text,
    )
