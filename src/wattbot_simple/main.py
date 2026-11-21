# 命令行入口，先补齐论文索引，再执行批量预测。

from wattbot_simple.index import build_chroma_index, PAPERS_DIR
from wattbot_simple.predict import parse_args, predict


# 将补齐后的索引直接交给本次批量预测复用。
def main():
    args = parse_args()
    chroma_store = build_chroma_index(PAPERS_DIR)
    predict(
        args.questions, args.output, ids=args.ids,
        chroma_store=chroma_store, thinking=args.thinking,
    )


# 支持通过 python -m wattbot_simple.main 运行建库和预测流程。
if __name__ == "__main__":
    main()
