from wattbot_simple.index import build_chroma_index, PAPERS_DIR
from wattbot_simple.predict import parse_args, predict


# 按命令行参数先补齐论文索引，再执行批量预测。
def main():
    args = parse_args(include_index_options=True)
    chroma_store = build_chroma_index(PAPERS_DIR, describe_pictures=args.describe_pictures)
    predict(
        args.questions, args.output, ids=args.ids,
        chroma_store=chroma_store, thinking=args.thinking,
    )


if __name__ == "__main__":
    main()
