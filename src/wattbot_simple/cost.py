import json
from contextlib import contextmanager
from datetime import datetime
from time import perf_counter

from langchain_core.callbacks import get_usage_metadata_callback


# 固定使用 DeepSeek Flash 高峰时段价格，单位为元/百万词元。
# 2026-11-15 核对：https://api-docs.deepseek.com/zh-cn/quick_start/pricing/
CACHE_HIT_PRICE = 0.04
CACHE_MISS_PRICE = 2.0
OUTPUT_PRICE = 8.0
TOTAL_FIELDS = (
    "elapsed_seconds", "input_tokens", "cache_hit_tokens", "cache_miss_tokens",
    "output_tokens", "total_tokens", "cost_cny",
)


# 记录一次答题尝试的用时和模型用量，成功或失败都会保存并累计成本。
@contextmanager
def record_cost(cost_path, question_id):
    # 续跑沿用已有账目；损坏的文件直接报错，避免覆盖历史成本。
    if cost_path.exists():
        with cost_path.open(encoding="utf-8") as file:
            cost = json.load(file)
    else:
        cost = {
            "records": [],
            "total": dict.fromkeys(TOTAL_FIELDS, 0),
        }
        cost["total"]["unknown_usage_count"] = 0

    # 保存开始时间，单题用时覆盖检索和回答，不计账目读写。
    started_at = datetime.now().astimezone()
    record = {
        "id": question_id,
        "started_at": started_at.isoformat(timespec="seconds"),
        "status": "failed",
        "model": None,
        **dict.fromkeys(TOTAL_FIELDS),
    }
    started = perf_counter()

    # 现成回调保留解析答案前的用量，因此答案格式错误也能记下已发生的消耗。
    with get_usage_metadata_callback() as callback:
        try:
            yield
            record["status"] = "success"
        finally:
            record["elapsed_seconds"] = round(perf_counter() - started, 3)
            _save_cost_record(cost_path, cost, record, callback.usage_metadata)


def _save_cost_record(cost_path, cost, record, usage_metadata):
    # 当前每题只调用一个 DeepSeek 模型，直接使用接口返回的实际词元数。
    if usage_metadata:
        model, usage = next(iter(usage_metadata.items()))
        input_tokens = usage["input_tokens"]
        output_tokens = usage["output_tokens"]
        # 回调的 cache_read 对应接口返回的命中数，剩余输入即未命中数。
        hit_tokens = usage["input_token_details"]["cache_read"]
        miss_tokens = input_tokens - hit_tokens
        record.update({
            "model": model,
            "input_tokens": input_tokens,
            "cache_hit_tokens": hit_tokens,
            "cache_miss_tokens": miss_tokens,
            "output_tokens": output_tokens,
            "total_tokens": usage["total_tokens"],
            "cost_cny": round((
                hit_tokens * CACHE_HIT_PRICE
                + miss_tokens * CACHE_MISS_PRICE
                + output_tokens * OUTPUT_PRICE
            ) / 1_000_000, 8),
        })
    else:
        # 超时等情况没有用量回执，保留空值并标记，不能当作零费用。
        cost["total"]["unknown_usage_count"] += 1

    # 每次实际尝试都计入累计值；同一题失败后重试会保留两条成本记录。
    cost["records"].append(record)
    total = cost["total"]
    for field in TOTAL_FIELDS:
        if record[field] is not None:
            total[field] = round(total[field] + record[field], 8)

    # 先完整写入临时文件，再替换账目，避免中断时留下半份 JSON。
    temporary_path = cost_path.with_suffix(cost_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(cost, file, ensure_ascii=False, indent=2)
        file.write("\n")
    temporary_path.replace(cost_path)

    # 显示本题与累计成本，累计费用仅包含已收到用量的尝试。
    if record["total_tokens"] is None:
        print(
            f"本题用时 {record['elapsed_seconds']:.3f} 秒，"
            "词元和费用未知（未收到用量）", flush=True,
        )
    else:
        print(
            f"本题用时 {record['elapsed_seconds']:.3f} 秒，"
            f"词元 {record['total_tokens']}，"
            f"高峰估算费用 {record['cost_cny']:.8f} 元", flush=True,
        )
    print(
        f"累计用时 {total['elapsed_seconds']:.3f} 秒，"
        f"已知词元 {total['total_tokens']}，"
        f"已知高峰估算费用 {total['cost_cny']:.8f} 元，"
        f"用量未知 {total['unknown_usage_count']} 次", flush=True,
    )
