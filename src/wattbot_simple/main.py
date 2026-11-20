# 命令行入口，先补齐论文索引，再执行批量预测。

from wattbot_simple.index import build_chroma_index, PAPERS_DIR
from wattbot_simple.predict import parse_args, predict


def main():
    args = parse_args()
    chroma_store = build_chroma_index(PAPERS_DIR)
    predict(
        args.questions, args.output, ids=args.ids,
        chroma_store=chroma_store, thinking=args.thinking,
    )


if __name__ == "__main__":
    main()
