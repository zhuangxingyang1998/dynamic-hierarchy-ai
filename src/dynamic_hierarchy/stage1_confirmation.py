"""Fail-closed cross-training-seed confirmation for revised Stage 1."""

from __future__ import annotations

import copy
import math
import statistics
from functools import lru_cache
from pathlib import Path
from typing import Any

from .stage1_config import (
    DEVELOPMENT_EVALUATION_SEEDS,
    stage1_config_digest,
    validated_experiment_compatibility_spec_digest,
    validated_experiment_spec_digest,
)
from .stage1_integrity import verify_result_manifests


def _stable_config(config: dict[str, Any]) -> dict[str, Any]:
    stable = copy.deepcopy(config)
    stable.pop("seed", None)
    return stable


def _range(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def run_completion_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Locate completion fields without pretending legacy results use schema 3."""
    schema_version = int(result.get("schema_version", -1))
    config = result.get("config", {})
    if schema_version >= 3:
        actual_step = result.get("global_step")
        target_step = result.get("target_steps")
        actual_step_field = "global_step"
        target_step_field = "target_steps"
        eligibility_explicit = "run_eligible_for_aggregation" in result
        aggregation_eligible = result.get("run_eligible_for_aggregation")
    else:
        actual_step = result.get("checkpoint_recovery", {}).get("current_step")
        target_step = config.get("optimizer_steps")
        actual_step_field = "checkpoint_recovery.current_step"
        target_step_field = "config.optimizer_steps"
        eligibility_explicit = False
        aggregation_eligible = None
    target_completion_observed = (
        result.get("state") == "completed"
        and result.get("reason") == "target_steps_reached"
        and actual_step is not None
        and target_step is not None
        and int(actual_step) == int(target_step)
    )
    return {
        "schema_version": schema_version,
        "actual_step": actual_step,
        "actual_step_field": actual_step_field,
        "target_step": target_step,
        "target_step_field": target_step_field,
        "aggregation_eligibility_explicit": eligibility_explicit,
        "aggregation_eligible": aggregation_eligible,
        "target_completion_observed": target_completion_observed,
    }


def run_completion_checks(result: dict[str, Any]) -> dict[str, bool]:
    config = result.get("config", {})
    metrics = result.get("metrics", {})
    evidence = run_completion_evidence(result)
    configured_target = int(config.get("optimizer_steps", -1))
    actual_step = evidence["actual_step"]
    target_step = evidence["target_step"]
    effective_batch = int(config.get("microbatch_size", 0)) * int(
        config.get("gradient_accumulation", 0)
    )
    models = metrics.get("models", {})
    model_names = ("A", "D_true", "D_sham")
    return {
        "result_schema_recognized": int(evidence["schema_version"]) >= 2,
        "state_completed": result.get("state") == "completed",
        "target_reason": result.get("reason") == "target_steps_reached",
        "exact_observed_step": (
            actual_step is not None
            and int(actual_step) == configured_target
        ),
        "target_steps_match": (
            target_step is not None
            and int(target_step) == configured_target
        ),
        "curriculum_complete": (
            metrics.get("curriculum_position", {}).get("complete") is True
        ),
        "all_model_updates_complete": all(
            int(models.get(name, {}).get("optimizer_updates", -1))
            == configured_target
            for name in model_names
        ),
        "all_model_examples_complete": all(
            int(models.get(name, {}).get("examples", -1))
            == configured_target * effective_batch
            for name in model_names
        ),
        "run_marked_eligible": evidence["aggregation_eligible"] is True,
    }


def _paired_split_effect(
    result: dict[str, Any],
    split_name: str,
) -> tuple[dict[str, float], bool]:
    config = result["config"]
    expected_examples = int(config["final_eval_examples_per_seed"])
    expected_eval_seeds = {str(seed) for seed in config["eval_seeds"]}
    split = result["final_evaluation"]["splits"][split_name]
    seeds = split.get("seeds", {})
    if set(seeds) != expected_eval_seeds:
        return {}, False
    d_minus_a = 0
    d_minus_sham = 0
    total = 0
    for seed_result in seeds.values():
        paired = seed_result.get("paired_sample_data")
        if not isinstance(paired, dict) or paired.get("schema_version") != 1:
            return {}, False
        masks = paired.get("correctness_masks", [])
        if (
            int(paired.get("sample_count", -1)) != expected_examples
            or len(masks) != expected_examples
            or int(seed_result.get("content_hash_count", -1)) != expected_examples
            or paired.get("content_hash_digest")
            != seed_result.get("content_hash_digest")
            or any(not isinstance(mask, int) or not 0 <= mask <= 7 for mask in masks)
        ):
            return {}, False
        a_correct = sum(mask & 1 for mask in masks)
        d_correct = sum((mask >> 1) & 1 for mask in masks)
        sham_correct = sum((mask >> 2) & 1 for mask in masks)
        models = seed_result.get("models", {})
        if any(
            int(models.get(name, {}).get("correct", -1)) != count
            for name, count in (
                ("A", a_correct),
                ("D_true", d_correct),
                ("D_sham", sham_correct),
            )
        ):
            return {}, False
        d_minus_a += d_correct - a_correct
        d_minus_sham += d_correct - sham_correct
        total += expected_examples
    return {
        "D_true_minus_A": d_minus_a / total,
        "D_true_minus_D_sham": d_minus_sham / total,
    }, True


def _student_t_pdf(value: float, degrees_of_freedom: int) -> float:
    numerator = math.gamma((degrees_of_freedom + 1) / 2)
    denominator = math.sqrt(degrees_of_freedom * math.pi) * math.gamma(
        degrees_of_freedom / 2
    )
    return numerator / denominator * (
        1 + value * value / degrees_of_freedom
    ) ** (-(degrees_of_freedom + 1) / 2)


def _adaptive_simpson(
    function,
    left: float,
    right: float,
    tolerance: float,
    whole: float,
    depth: int,
) -> float:
    middle = (left + right) / 2
    left_middle = (left + middle) / 2
    right_middle = (middle + right) / 2
    left_value = (
        middle - left
    ) / 6 * (
        function(left)
        + 4 * function(left_middle)
        + function(middle)
    )
    right_value = (
        right - middle
    ) / 6 * (
        function(middle)
        + 4 * function(right_middle)
        + function(right)
    )
    delta = left_value + right_value - whole
    if depth <= 0 or abs(delta) <= 15 * tolerance:
        return left_value + right_value + delta / 15
    return _adaptive_simpson(
        function,
        left,
        middle,
        tolerance / 2,
        left_value,
        depth - 1,
    ) + _adaptive_simpson(
        function,
        middle,
        right,
        tolerance / 2,
        right_value,
        depth - 1,
    )


def _student_t_cdf(value: float, degrees_of_freedom: int) -> float:
    if value == 0:
        return 0.5
    sign = 1 if value > 0 else -1
    upper = abs(value)
    function = lambda item: _student_t_pdf(item, degrees_of_freedom)
    whole = upper / 6 * (
        function(0.0) + 4 * function(upper / 2) + function(upper)
    )
    integral = _adaptive_simpson(
        function,
        0.0,
        upper,
        1e-10,
        whole,
        20,
    )
    return 0.5 + sign * integral


@lru_cache(maxsize=None)
def _student_t_quantile(probability: float, degrees_of_freedom: int) -> float:
    if not 0.5 < probability < 1.0:
        raise ValueError("one-sided t probability must be between 0.5 and one")
    lower = 0.0
    upper = 1.0
    while _student_t_cdf(upper, degrees_of_freedom) < probability:
        upper *= 2
    for _ in range(64):
        middle = (lower + upper) / 2
        if _student_t_cdf(middle, degrees_of_freedom) < probability:
            lower = middle
        else:
            upper = middle
    return (lower + upper) / 2


def _one_sided_paired_t_lower_bound(
    values: list[float],
    alpha: float,
) -> dict[str, float | int]:
    if len(values) < 2:
        raise ValueError("paired training-seed confidence interval requires two seeds")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    standard_error = standard_deviation / math.sqrt(len(values))
    critical = _student_t_quantile(1 - alpha, len(values) - 1)
    return {
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "critical_t": critical,
        "one_sided_lower_bound": mean - critical * standard_error,
        "training_seed_count": len(values),
    }


def aggregate_confirmation(
    results: list[dict[str, Any]],
    result_paths: list[Path] | None = None,
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one result is required")
    if result_paths is not None and len(result_paths) != len(results):
        raise ValueError("result_paths must align exactly with results")
    first_config = results[0]["config"]
    expected_seeds = tuple(int(seed) for seed in first_config["confirmation_training_seeds"])
    minimum_seeds = int(first_config["minimum_confirmation_training_seeds"])
    stable_config = _stable_config(first_config)
    source_manifest_hash = results[0].get("manifest", {}).get("manifest_hash")
    snapshot_manifest_hash = results[0].get("snapshot_manifest_hash")
    split_specs = {
        item["name"]: item for item in first_config["evaluation_splits"]
    }
    expected_split_names = tuple(split_specs)

    observed_seeds: list[int] = []
    run_checks: dict[str, dict[str, bool]] = {}
    run_effects: dict[int, dict[str, dict[str, float]]] = {}
    for result_index, result in enumerate(results):
        config = result.get("config", {})
        seed = int(config.get("seed", -1))
        observed_seeds.append(seed)
        completion = run_completion_checks(result)
        split_effects: dict[str, dict[str, float]] = {}
        paired_data_complete = True
        if tuple(result.get("final_evaluation", {}).get("splits", {})) != expected_split_names:
            paired_data_complete = False
        else:
            for split_name in expected_split_names:
                effect, valid = _paired_split_effect(result, split_name)
                paired_data_complete &= valid
                if valid:
                    split_effects[split_name] = effect
        run_effects[seed] = split_effects
        prerequisite = result.get("candidate_prerequisite", {})
        prerequisite_expected = prerequisite.get("expected", {})
        pinned_prerequisite = {
            "config_digest": config.get("candidate_prerequisite_config_digest"),
            "manifest_hash": config.get("candidate_prerequisite_manifest_hash"),
            "snapshot_manifest_hash": config.get(
                "candidate_prerequisite_snapshot_manifest_hash"
            ),
            "result_digest": config.get("candidate_prerequisite_result_digest"),
            "experiment_spec_digest": config.get(
                "candidate_prerequisite_experiment_spec_digest"
            ),
            "compatibility_spec_digest": config.get(
                "candidate_prerequisite_compatibility_spec_digest"
            ),
        }
        final_evaluation = result.get("final_evaluation", {})
        manifest_checks = (
            verify_result_manifests(
                result,
                result_paths[result_index].resolve().parent,
            )
            if result_paths is not None
            else {"manifest_result_path_provided": False}
        )
        try:
            run_experiment_spec_digest = validated_experiment_spec_digest(config)
            run_compatibility_spec_digest = (
                validated_experiment_compatibility_spec_digest(config)
            )
            experiment_spec_valid = True
        except (TypeError, ValueError):
            run_experiment_spec_digest = None
            run_compatibility_spec_digest = None
            experiment_spec_valid = False
        run_checks[str(seed)] = {
            **completion,
            **manifest_checks,
            "current_result_schema": int(result.get("schema_version", -1)) >= 3,
            "explicit_aggregation_eligibility": (
                "run_eligible_for_aggregation" in result
                and result.get("run_eligible_for_aggregation") is True
            ),
            "formal_evaluation": config.get("formal_evaluation") is True,
            "evaluation_scale": int(config.get("final_eval_examples_per_seed", 0))
            >= 10_000,
            "multiple_unused_evaluation_seeds": (
                len(config.get("eval_seeds", ())) >= 2
                and not set(config.get("eval_seeds", ()))
                & DEVELOPMENT_EVALUATION_SEEDS
            ),
            "final_evaluation_spec_matches_config": (
                int(final_evaluation.get("examples_per_split_seed", -1))
                == int(config.get("final_eval_examples_per_seed", -2))
                and tuple(final_evaluation.get("evaluation_seeds", ()))
                == tuple(config.get("eval_seeds", ()))
                and final_evaluation.get("kind") == "formal_confirmation"
            ),
            "final_content_and_shape_audits_pass": (
                final_evaluation.get("overlap_audit", {}).get(
                    "all_content_disjoint"
                )
                is True
                and final_evaluation.get("overlap_audit", {}).get(
                    "all_shape_rules_valid"
                )
                is True
            ),
            "config_digest_matches": result.get("config_digest")
            == stage1_config_digest(config),
            "validated_experiment_spec_digest_matches": (
                experiment_spec_valid
                and result.get("validated_experiment_spec_digest")
                == run_experiment_spec_digest
            ),
            "validated_experiment_compatibility_spec_digest_matches": (
                experiment_spec_valid
                and result.get(
                    "validated_experiment_compatibility_spec_digest"
                )
                == run_compatibility_spec_digest
            ),
            "formal_compatibility_spec_matches_candidate_pin": (
                run_compatibility_spec_digest
                == pinned_prerequisite["compatibility_spec_digest"]
            ),
            "candidate_prerequisite_pins_present": all(
                pinned_prerequisite.values()
            ),
            "candidate_prerequisite_verified": (
                prerequisite.get("required") is True
                and prerequisite.get("passed") is True
            )
            and prerequisite_expected == pinned_prerequisite,
            "candidate_pass": (
                result.get("candidate_gate", {}).get("candidate_pass") is True
            ),
            "formal_final_attempt_completed": (
                result.get("formal_final_attempt", {}).get("required") is True
                and result.get("formal_final_attempt", {}).get("state")
                == "completed"
            ),
            "single_run_did_not_unblock_stage2": (
                result.get("candidate_gate", {}).get("stage2_unblocked") is False
            ),
            "same_config_except_training_seed": _stable_config(config) == stable_config,
            "same_source_manifest": (
                isinstance(source_manifest_hash, str)
                and bool(source_manifest_hash)
                and result.get("manifest", {}).get("manifest_hash")
                == source_manifest_hash
            ),
            "same_snapshot_manifest": (
                isinstance(snapshot_manifest_hash, str)
                and bool(snapshot_manifest_hash)
                and result.get("snapshot_manifest_hash") == snapshot_manifest_hash
            ),
            "paired_sample_data_complete": paired_data_complete,
        }

    unique_observed = set(observed_seeds)
    conditions = {
        "minimum_training_seed_count": len(unique_observed) >= minimum_seeds,
        "no_duplicate_training_seeds": len(unique_observed) == len(observed_seeds),
        "exact_declared_training_seed_set": unique_observed == set(expected_seeds),
        "all_runs_pass_integrity_and_per_run_gates": all(
            all(checks.values()) for checks in run_checks.values()
        ),
    }

    split_aggregates: dict[str, object] = {}
    statistical_conditions: dict[str, bool] = {}
    statistical_gate_passed = False
    if all(conditions.values()):
        comparisons = len(expected_split_names) * 2
        familywise_alpha = float(first_config["confirmation_familywise_alpha"])
        corrected_alpha = familywise_alpha / comparisons
        for split_name in expected_split_names:
            effects: dict[str, object] = {}
            category = split_specs[split_name]["category"]
            for effect_name in ("D_true_minus_A", "D_true_minus_D_sham"):
                values = [
                    run_effects[seed][split_name][effect_name]
                    for seed in expected_seeds
                ]
                interval = _one_sided_paired_t_lower_bound(values, corrected_alpha)
                if effect_name == "D_true_minus_A":
                    threshold = (
                        float(first_config["gate"]["minimum_d_advantage_in_distribution"])
                        if category == "in_distribution"
                        else float(first_config["gate"]["minimum_d_advantage_extrapolation"])
                    )
                else:
                    threshold = (
                        float(first_config["gate"]["minimum_d_over_sham_in_distribution"])
                        if category == "in_distribution"
                        else float(first_config["gate"]["minimum_d_over_sham_extrapolation"])
                    )
                passed = float(interval["one_sided_lower_bound"]) >= threshold
                statistical_conditions[f"{split_name}:{effect_name}"] = passed
                effects[effect_name] = {
                    "training_seed_effects": values,
                    "interval": interval,
                    "threshold": threshold,
                    "passed": passed,
                }
            split_aggregates[split_name] = {
                "category": category,
                "paired_effects": effects,
            }
        in_distribution_pass = all(
            value
            for name, value in statistical_conditions.items()
            if split_specs[name.split(":", 1)[0]]["category"] == "in_distribution"
        )
        extrapolation_pass = any(
            all(
                statistical_conditions[f"{split_name}:{effect_name}"]
                for effect_name in ("D_true_minus_A", "D_true_minus_D_sham")
            )
            for split_name, spec in split_specs.items()
            if spec["category"] != "in_distribution"
        )
        statistical_gate_passed = in_distribution_pass and extrapolation_pass
        conditions["statistical_all_in_distribution_effects"] = in_distribution_pass
        conditions["statistical_at_least_one_extrapolation_split"] = extrapolation_pass

    stage2_unblocked = all(conditions.values()) and statistical_gate_passed
    return {
        "schema_version": 2,
        "training_seeds": sorted(unique_observed),
        "required_training_seeds": list(expected_seeds),
        "minimum_training_seeds": minimum_seeds,
        "run_checks": run_checks,
        "conditions": conditions,
        "statistical_plan": {
            "unit": "independent training seed",
            "effect_inputs": "per-example paired correctness masks",
            "interval": "one-sided paired Student t lower confidence bound",
            "familywise_alpha": first_config["confirmation_familywise_alpha"],
            "multiplicity_correction": first_config[
                "confirmation_multiplicity_correction"
            ],
            "comparison_count": len(expected_split_names) * 2,
            "corrected_alpha": (
                float(first_config["confirmation_familywise_alpha"])
                / (len(expected_split_names) * 2)
            ),
        },
        "statistical_conditions": statistical_conditions,
        "split_aggregates": split_aggregates,
        "stage2_unblocked": stage2_unblocked,
        "decision": (
            "formal_confirmation_passed"
            if stage2_unblocked
            else "formal_confirmation_incomplete_or_failed"
        ),
    }
