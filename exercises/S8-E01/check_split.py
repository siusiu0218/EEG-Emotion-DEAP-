"""S8-E01 learner scaffold: check subject IDs before any training."""


def validate_subject_split(train_ids, val_ids, test_ids):
    # TODO 1: 三组均非空，每组内部没有重复；不满足时raise ValueError。
    # TODO 2: 转为set，检查任意两组没有交集。
    # TODO 3: 检查三组并集恰好等于set(range(1, 33))。
    raise NotImplementedError("请补全受试者名单检查")


if __name__ == "__main__":
    train = list(range(1, 23))
    val = list(range(23, 28))
    test = list(range(28, 33))
    assert validate_subject_split(train, val, test) is True
    print("正常名单：通过")

    invalid_cases = [
        ("两组交叉", train, val, [27, 28, 29, 30, 31, 32]),
        ("漏掉32号", train, val, [28, 29, 30, 31]),
        ("组内重复", train + [1], val, test),
        ("空验证组", train, [], val + test),
    ]
    for name, train_ids, val_ids, test_ids in invalid_cases:
        try:
            validate_subject_split(train_ids, val_ids, test_ids)
        except ValueError:
            print(f"{name}：已拒绝")
        else:
            raise AssertionError(f"没有识别非法名单：{name}")
    print("S8-E01_CASES_PASSED")
