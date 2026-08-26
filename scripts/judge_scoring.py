"""Deterministic benchmark scores derived from Judge dimension grades."""


GRADE_SCORE = {"G": 1.0, "S": 0.5, "B": 0.0}


def _applicable_scores(prediction, task_spec):
    dimensions = prediction.get("dimensions")
    if not isinstance(dimensions, dict):
        return None

    scores = {}
    for name, spec in task_spec.get("dimensions", {}).items():
        item = dimensions.get(name)
        if not isinstance(item, dict):
            if spec.get("applicable"):
                return None
            continue
        if not spec.get("applicable"):
            item["score"] = None
            continue
        grade = item.get("grade")
        if grade not in GRADE_SCORE:
            return None
        item["score"] = GRADE_SCORE[grade]
        scores[name] = GRADE_SCORE[grade]
    return scores


def _coupled_item_score(scores, *, omit_d3=False):
    """Return the item score after coupling applicable D1/D2 by their minimum."""
    components = []
    interaction_scores = [
        scores[name] for name in ("D1", "D2") if name in scores
    ]
    if interaction_scores:
        components.append(min(interaction_scores))
    components.extend(
        scores[name]
        for name in ("D3", "D4", "D5")
        if name in scores and not (omit_d3 and name == "D3")
    )
    return sum(components) / len(components) if components else None


def apply_dimension_mean_score(prediction, task_spec):
    """Normalize scores and attach both diagnostic and official item scores.

    The language model chooses only G/S/B. Numeric scores are always derived
    here so prompt arithmetic and workbook weights cannot affect the result.
    The return value remains the raw applicable-dimension mean for compatibility;
    ``benchmark_score`` is the official item-level D1/D2-coupled score.
    """
    if not isinstance(prediction, dict):
        return None

    scores = _applicable_scores(prediction, task_spec)
    if not scores:
        return None

    dimension_mean = sum(scores.values()) / len(scores)
    benchmark_score = _coupled_item_score(scores)
    score_without_d3 = _coupled_item_score(scores, omit_d3=True)
    prediction["dimension_mean_score"] = dimension_mean
    prediction["benchmark_score"] = benchmark_score
    prediction["score_without_d3"] = score_without_d3
    overall = prediction.get("overall")
    if isinstance(overall, dict):
        overall.pop("weighted_score", None)
        overall["dimension_mean_score"] = dimension_mean
        overall["benchmark_score"] = benchmark_score
        overall["score_without_d3"] = score_without_d3
        overall["max_score"] = 1.0
    return dimension_mean
