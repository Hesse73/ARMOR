try:
    from math_verify import parse, verify
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")


def compute_score(solution_str: str, ground_truth: str):
    gt_parsed = parse(ground_truth)
    solution_str = solution_str[-300:]
    solution_parsed = parse(solution_str)
    correct = verify(gt_parsed, solution_parsed)
    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": solution_str,
    }