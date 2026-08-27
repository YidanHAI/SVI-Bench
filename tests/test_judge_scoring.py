from judge_scoring import apply_dimension_mean_score


def dimension(grade):
    return {"grade": grade}


def spec(*applicable):
    selected = set(applicable)
    return {
        "dimensions": {
            name: {"applicable": name in selected}
            for name in ("D1", "D2", "D3", "D4", "D5")
        }
    }


def test_official_score_couples_d1_d2_within_each_item():
    prediction = {
        "dimensions": {
            "D1": dimension("B"),
            "D2": dimension("G"),
            "D3": dimension("S"),
            "D4": dimension("G"),
            "D5": dimension("not_applicable"),
        },
        "overall": {"weighted_score": 0.99},
    }

    diagnostic_mean = apply_dimension_mean_score(
        prediction, spec("D1", "D2", "D3", "D4")
    )

    assert diagnostic_mean == 0.625
    assert prediction["benchmark_score"] == 0.5
    assert prediction["score_without_d3"] == 0.5
    assert prediction["overall"] == {
        "dimension_mean_score": 0.625,
        "benchmark_score": 0.5,
        "score_without_d3": 0.5,
        "max_score": 1.0,
    }


def test_single_applicable_d1_or_d2_is_retained():
    prediction = {
        "dimensions": {
            "D1": dimension("G"),
            "D2": dimension("not_applicable"),
            "D3": dimension("B"),
            "D4": dimension("S"),
            "D5": dimension("not_applicable"),
        }
    }

    apply_dimension_mean_score(prediction, spec("D1", "D3", "D4"))

    assert prediction["benchmark_score"] == 0.5
    assert prediction["score_without_d3"] == 0.75


def test_missing_applicable_grade_fails_closed():
    prediction = {"dimensions": {"D1": dimension("G")}}

    assert apply_dimension_mean_score(prediction, spec("D1", "D2")) is None
    assert "benchmark_score" not in prediction
