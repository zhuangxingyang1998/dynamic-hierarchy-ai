"""Configuration contracts for the original and revised Stage 1 experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import DataConfig, ModelConfig, _nonnegative_int, _positive_int


TOPOLOGIES = {"leaf", "skew", "balanced", "branched"}
DEVELOPMENT_EVALUATION_SEEDS = {
    11003,
    22003,
    33013,
    44017,
    92041,
    92051,
    92063,
}
BASELINE_GATE_POLICIES = {
    "joint_all_required_v1",
    "privileged_structure_posthoc_v1",
}
EXPERIMENT_SPEC_ALLOWED_VARIATION_FIELDS = {
    "seed",
    "device",
    "time_budget_minutes",
    "cpu_threads",
    "deterministic",
    "yield_ms",
    "eval_batches",
    "eval_interval_steps",
    "heartbeat_examples",
    "final_eval_examples_per_seed",
    "final_eval_batch_size",
    "eval_seeds",
    "formal_evaluation",
    "requires_candidate_pass",
    "candidate_prerequisite_config_digest",
    "candidate_prerequisite_manifest_hash",
    "candidate_prerequisite_snapshot_manifest_hash",
    "candidate_prerequisite_result_digest",
    "candidate_prerequisite_experiment_spec_digest",
    "candidate_prerequisite_result_path",
    "candidate_prerequisite_compatibility_spec_digest",
    "checkpoint_steps",
    "checkpoint_minutes",
    "heartbeat_seconds",
    "resource_sample_seconds",
    "cpu_pause_percent",
    "cpu_resume_percent",
    "ram_pause_gb",
    "ram_resume_gb",
    "pressure_samples",
    "recovery_samples",
}
EXPERIMENT_COMPATIBILITY_ALLOWED_VARIATION_FIELDS = (
    EXPERIMENT_SPEC_ALLOWED_VARIATION_FIELDS
    | {"confirmation_training_seeds", "foundation_eval_seed"}
)


def stage1_config_digest(config: dict[str, object]) -> str:
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class TrainingProfile:
    name: str
    depth: int
    topology: str

    def validate(self) -> None:
        if not self.name:
            raise ValueError("training profile name must not be empty")
        _nonnegative_int("training profile depth", self.depth)
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unsupported training topology: {self.topology}")
        if (self.depth == 0) != (self.topology == "leaf"):
            raise ValueError("depth zero requires leaf topology and leaf topology requires depth zero")
        if self.topology == "branched" and self.depth < 3:
            raise ValueError("branched topology requires depth at least three")


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    steps: int
    profiles: tuple[TrainingProfile, ...]

    def validate(self) -> None:
        if not self.name:
            raise ValueError("curriculum stage name must not be empty")
        _positive_int("curriculum stage steps", self.steps)
        if not self.profiles:
            raise ValueError(f"curriculum stage {self.name!r} must contain profiles")
        for profile in self.profiles:
            profile.validate()


@dataclass(frozen=True)
class EvaluationSplit:
    name: str
    depth: int
    topology: str
    category: str
    shape_partition: str
    required_above_majority: bool = True

    def validate(self) -> None:
        if not self.name:
            raise ValueError("evaluation split name must not be empty")
        _nonnegative_int("evaluation split depth", self.depth)
        if self.topology not in TOPOLOGIES:
            raise ValueError(f"unsupported evaluation topology: {self.topology}")
        if self.category not in {"in_distribution", "depth_extrapolation", "topology_extrapolation"}:
            raise ValueError(f"unsupported evaluation category: {self.category}")
        if self.shape_partition not in {"train", "heldout"}:
            raise ValueError("shape_partition must be 'train' or 'heldout'")
        if (self.depth == 0) != (self.topology == "leaf"):
            raise ValueError("depth zero requires leaf topology and leaf topology requires depth zero")
        if self.topology == "branched" and self.depth < 3:
            raise ValueError("branched topology requires depth at least three")
        if type(self.required_above_majority) is not bool:
            raise ValueError("required_above_majority must be a boolean")


@dataclass(frozen=True)
class GateConfig:
    minimum_d_advantage_in_distribution: float = 0.03
    minimum_d_advantage_extrapolation: float = 0.02
    minimum_d_over_sham_in_distribution: float = 0.03
    minimum_d_over_sham_extrapolation: float = 0.02
    minimum_above_majority: float = 0.03
    require_all_in_distribution_splits: bool = True
    baseline_policy: str = "joint_all_required_v1"

    def validate(self) -> None:
        for name in (
            "minimum_d_advantage_in_distribution",
            "minimum_d_advantage_extrapolation",
            "minimum_d_over_sham_in_distribution",
            "minimum_d_over_sham_extrapolation",
            "minimum_above_majority",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between zero and one")
        if type(self.require_all_in_distribution_splits) is not bool:
            raise ValueError("require_all_in_distribution_splits must be a boolean")
        if self.require_all_in_distribution_splits is not True:
            raise ValueError("revised Stage 1 conservatively requires every in-distribution split")
        if self.baseline_policy not in BASELINE_GATE_POLICIES:
            raise ValueError(
                f"baseline_policy must be one of {sorted(BASELINE_GATE_POLICIES)}"
            )


def _default_curriculum() -> tuple[CurriculumStage, ...]:
    return (
        CurriculumStage("binding_lookup", 100, (TrainingProfile("lookup", 0, "leaf"),)),
        CurriculumStage("depth_1", 100, (TrainingProfile("depth1_skew", 1, "skew"),)),
        CurriculumStage(
            "depth_2",
            150,
            (
                TrainingProfile("depth2_skew", 2, "skew"),
                TrainingProfile("depth2_balanced", 2, "balanced"),
            ),
        ),
        CurriculumStage(
            "depth_3",
            250,
            (
                TrainingProfile("depth3_skew", 3, "skew"),
                TrainingProfile("depth3_balanced", 3, "balanced"),
            ),
        ),
        CurriculumStage(
            "mixed_consolidation",
            400,
            (
                TrainingProfile("lookup", 0, "leaf"),
                TrainingProfile("depth1_skew", 1, "skew"),
                TrainingProfile("depth2_skew", 2, "skew"),
                TrainingProfile("depth2_balanced", 2, "balanced"),
                TrainingProfile("depth3_skew", 3, "skew"),
                TrainingProfile("depth3_balanced", 3, "balanced"),
            ),
        ),
    )


def _default_splits() -> tuple[EvaluationSplit, ...]:
    return (
        EvaluationSplit("id_depth3_skew", 3, "skew", "in_distribution", "train"),
        EvaluationSplit("id_depth3_balanced", 3, "balanced", "in_distribution", "train"),
        EvaluationSplit("depth5_skew", 5, "skew", "depth_extrapolation", "heldout"),
        EvaluationSplit(
            "heldout_shape_depth3_branched",
            3,
            "branched",
            "topology_extrapolation",
            "heldout",
        ),
    )


@dataclass(frozen=True)
class Stage1Config:
    revision: str = "pilot-v1"
    operand_mode: str = "bound_variable"
    seed: int = 7301
    device: str = "directml"
    optimizer_steps: int = 100_000
    time_budget_minutes: float = 120.0
    microbatch_size: int = 4
    gradient_accumulation: int = 4
    learning_rate: float = 0.001
    cpu_threads: int = 2
    deterministic: bool = False
    yield_ms: int = 100
    eval_batches: int = 4
    eval_length_scales: tuple[int, ...] = (1, 2, 4)
    eval_interval_steps: int = 100
    heartbeat_examples: int = 28
    final_eval_examples_per_seed: int = 1029
    final_eval_batch_size: int = 49
    eval_seeds: tuple[int, ...] = (11003, 22003, 33013)
    confirmation_training_seeds: tuple[int, ...] = (
        7301,
        7307,
        7321,
        7331,
        7333,
        7349,
        7351,
        7369,
    )
    minimum_confirmation_training_seeds: int = 8
    formal_evaluation: bool = False
    requires_candidate_pass: bool = False
    candidate_prerequisite_config_digest: str = ""
    candidate_prerequisite_manifest_hash: str = ""
    candidate_prerequisite_snapshot_manifest_hash: str = ""
    candidate_prerequisite_result_digest: str = ""
    candidate_prerequisite_experiment_spec_digest: str = ""
    candidate_prerequisite_result_path: str = ""
    candidate_prerequisite_compatibility_spec_digest: str = ""
    confirmation_familywise_alpha: float = 0.05
    confirmation_ci_method: str = "paired_training_seed_t"
    confirmation_multiplicity_correction: str = "bonferroni"
    foundation_gate_required: bool = False
    foundation_eval_examples: int = 700
    foundation_eval_batch_size: int = 70
    foundation_eval_seed: int = 44017
    foundation_c0_min_accuracy: float = 0.99
    foundation_c1_min_accuracy: float = 0.98
    max_generation_attempts_per_example: int = 256
    max_evaluation_generation_attempts_per_example: int = 256
    checkpoint_steps: int = 100
    checkpoint_minutes: float = 5.0
    heartbeat_seconds: float = 20.0
    resource_sample_seconds: float = 5.0
    cpu_pause_percent: float = 85.0
    cpu_resume_percent: float = 75.0
    ram_pause_gb: float = 6.0
    ram_resume_gb: float = 8.0
    pressure_samples: int = 3
    recovery_samples: int = 2
    tasks: tuple[str, ...] = ("nested_expression",)
    training_topologies: tuple[str, ...] = ("skew", "balanced")
    held_out_topologies: tuple[str, ...] = ("branched",)
    curriculum: tuple[CurriculumStage, ...] = field(default_factory=_default_curriculum)
    evaluation_splits: tuple[EvaluationSplit, ...] = field(default_factory=_default_splits)
    gate: GateConfig = field(default_factory=GateConfig)
    data: DataConfig = field(
        default_factory=lambda: DataConfig(
            vocab_size=64,
            repeat_length=6,
            binding_pairs=4,
            expression_depth=3,
            expression_variables=6,
            expression_values=7,
        )
    )
    model_a: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            embedding_dim=64,
            heads=4,
            layers=2,
            feedforward_dim=128,
            dropout=0.0,
        )
    )
    model_d: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            embedding_dim=64,
            heads=4,
            layers=1,
            feedforward_dim=128,
            dropout=0.0,
        )
    )

    @property
    def effective_batch_size(self) -> int:
        return self.microbatch_size * self.gradient_accumulation

    @property
    def revised(self) -> bool:
        return self.revision == "revised-v2"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def validate(self) -> None:
        _positive_int("optimizer_steps", self.optimizer_steps)
        _positive_int("microbatch_size", self.microbatch_size)
        _positive_int("gradient_accumulation", self.gradient_accumulation)
        _positive_int("cpu_threads", self.cpu_threads)
        _nonnegative_int("yield_ms", self.yield_ms)
        _positive_int("eval_batches", self.eval_batches)
        _positive_int("eval_interval_steps", self.eval_interval_steps)
        _positive_int("heartbeat_examples", self.heartbeat_examples)
        _positive_int("final_eval_examples_per_seed", self.final_eval_examples_per_seed)
        _positive_int("final_eval_batch_size", self.final_eval_batch_size)
        _positive_int("max_generation_attempts_per_example", self.max_generation_attempts_per_example)
        _positive_int(
            "max_evaluation_generation_attempts_per_example",
            self.max_evaluation_generation_attempts_per_example,
        )
        _positive_int("minimum_confirmation_training_seeds", self.minimum_confirmation_training_seeds)
        _positive_int("foundation_eval_examples", self.foundation_eval_examples)
        _positive_int("foundation_eval_batch_size", self.foundation_eval_batch_size)
        _positive_int("checkpoint_steps", self.checkpoint_steps)
        _positive_int("pressure_samples", self.pressure_samples)
        _positive_int("recovery_samples", self.recovery_samples)
        if self.revision not in {"pilot-v1", "revised-v2"}:
            raise ValueError("revision must be 'pilot-v1' or 'revised-v2'")
        if self.operand_mode not in {"bound_variable", "literal"}:
            raise ValueError("operand_mode must be 'bound_variable' or 'literal'")
        if self.device not in {"cpu", "directml"}:
            raise ValueError("device must be 'cpu' or 'directml'")
        for name in (
            "deterministic",
            "formal_evaluation",
            "requires_candidate_pass",
            "foundation_gate_required",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a boolean")
        if self.device == "directml" and self.deterministic is True:
            raise ValueError("DirectML Stage 1 requires deterministic=false")
        for name, value in (
            ("time_budget_minutes", self.time_budget_minutes),
            ("learning_rate", self.learning_rate),
            ("checkpoint_minutes", self.checkpoint_minutes),
            ("heartbeat_seconds", self.heartbeat_seconds),
            ("resource_sample_seconds", self.resource_sample_seconds),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cpu_resume_percent >= self.cpu_pause_percent:
            raise ValueError("cpu_resume_percent must be lower than cpu_pause_percent")
        if self.ram_resume_gb <= self.ram_pause_gb:
            raise ValueError("ram_resume_gb must be greater than ram_pause_gb")
        if self.tasks != ("nested_expression",):
            raise ValueError("Stage 1 supports only nested_expression")
        if not self.eval_length_scales:
            raise ValueError("eval_length_scales must not be empty")
        for scale in self.eval_length_scales:
            _positive_int("eval_length_scales entries", scale)
        self.data.validate(max(self.eval_length_scales))
        self.model_a.validate()
        self.model_d.validate()
        if not self.revised:
            return

        if self.data.expression_values != 7:
            raise ValueError("revised Stage 1 requires prime modulus expression_values=7")
        if self.effective_batch_size % self.data.expression_values:
            raise ValueError("revised effective batch size must be divisible by expression_values")
        if self.operand_mode == "literal" and self.effective_batch_size % 8:
            raise ValueError("literal effective batch size must be divisible by both 7 and 8")
        if self.heartbeat_examples % self.data.expression_values:
            raise ValueError("heartbeat_examples must be divisible by expression_values")
        if self.final_eval_batch_size % self.data.expression_values:
            raise ValueError("final_eval_batch_size must be divisible by expression_values")
        if self.final_eval_examples_per_seed % self.final_eval_batch_size:
            raise ValueError("final_eval_examples_per_seed must be divisible by final_eval_batch_size")
        if self.foundation_eval_batch_size % self.data.expression_values:
            raise ValueError("foundation_eval_batch_size must be divisible by expression_values")
        if self.foundation_eval_examples % self.foundation_eval_batch_size:
            raise ValueError("foundation_eval_examples must be divisible by foundation_eval_batch_size")
        if self.foundation_gate_required and self.operand_mode != "literal":
            raise ValueError("the foundation gate is defined only for operand_mode='literal'")
        if self.foundation_gate_required and self.foundation_eval_examples < 700:
            raise ValueError("literal foundation gate requires at least 700 fixed examples per task")
        if self.foundation_gate_required and (
            self.foundation_c0_min_accuracy != 0.99
            or self.foundation_c1_min_accuracy != 0.98
        ):
            raise ValueError("literal foundation thresholds are fixed at C0=0.99 and C1=0.98")
        if self.formal_evaluation and self.final_eval_examples_per_seed < 10_000:
            raise ValueError("formal confirmation requires at least 10000 examples per split and seed")
        if len(self.eval_seeds) < (2 if self.formal_evaluation else 1):
            raise ValueError("formal evaluation requires multiple fixed evaluation seeds")
        if len(set(self.eval_seeds)) != len(self.eval_seeds):
            raise ValueError("evaluation seeds must be unique")
        if len(set(self.confirmation_training_seeds)) != len(self.confirmation_training_seeds):
            raise ValueError("confirmation training seeds must be unique")
        if self.formal_evaluation:
            all_formal_seeds = (
                *self.confirmation_training_seeds,
                *self.eval_seeds,
                self.foundation_eval_seed,
            )
            if len(set(all_formal_seeds)) != len(all_formal_seeds):
                raise ValueError(
                    "formal training, evaluation, and foundation seeds must be "
                    "pairwise disjoint"
                )
        if len(self.confirmation_training_seeds) < self.minimum_confirmation_training_seeds:
            raise ValueError("confirmation plan must preserve at least eight training seeds")
        if self.formal_evaluation and self.minimum_confirmation_training_seeds < 8:
            raise ValueError("formal confirmation requires at least eight training seeds")
        if self.formal_evaluation and self.operand_mode == "literal" and not self.requires_candidate_pass:
            raise ValueError("formal literal evaluation requires a prior candidate pass record")
        prerequisite_fields = (
            self.candidate_prerequisite_config_digest,
            self.candidate_prerequisite_manifest_hash,
            self.candidate_prerequisite_snapshot_manifest_hash,
            self.candidate_prerequisite_result_digest,
            self.candidate_prerequisite_experiment_spec_digest,
            self.candidate_prerequisite_result_path,
            self.candidate_prerequisite_compatibility_spec_digest,
        )
        if any(prerequisite_fields) and not all(prerequisite_fields):
            raise ValueError(
                "candidate prerequisite path and all identity/compatibility "
                "pins must be set together"
            )
        if not 0.0 < self.confirmation_familywise_alpha < 0.5:
            raise ValueError("confirmation_familywise_alpha must be between zero and 0.5")
        if self.confirmation_ci_method != "paired_training_seed_t":
            raise ValueError("confirmation_ci_method must be 'paired_training_seed_t'")
        if self.confirmation_multiplicity_correction != "bonferroni":
            raise ValueError("confirmation_multiplicity_correction must be 'bonferroni'")
        if self.formal_evaluation and set(self.eval_seeds) & DEVELOPMENT_EVALUATION_SEEDS:
            raise ValueError("formal evaluation seeds must be unused by development evaluation")
        if self.seed not in self.confirmation_training_seeds:
            raise ValueError("the run seed must belong to confirmation_training_seeds")
        if not self.curriculum:
            raise ValueError("revised curriculum must not be empty")
        expected_stage_names = (
            (
                "literal_c0",
                "literal_c1",
                "literal_depth_2",
                "literal_depth_3",
                "literal_rehearsal",
            )
            if self.operand_mode == "literal"
            else (
                "binding_lookup",
                "depth_1",
                "depth_2",
                "depth_3",
                "mixed_consolidation",
            )
        )
        if tuple(stage.name for stage in self.curriculum) != expected_stage_names:
            raise ValueError(f"revised curriculum stages must be exactly {expected_stage_names}")
        for stage in self.curriculum:
            stage.validate()
        if sum(stage.steps for stage in self.curriculum) != self.optimizer_steps:
            raise ValueError("optimizer_steps must equal the sum of revised curriculum stage steps")
        observed_depths = {
            profile.depth
            for stage in self.curriculum[:-1]
            for profile in stage.profiles
        }
        if observed_depths != {0, 1, 2, 3}:
            raise ValueError("revised curriculum must explicitly cover binding lookup and depths 1, 2, 3")
        if self.operand_mode == "literal":
            if any(profile.depth != 0 for profile in self.curriculum[0].profiles):
                raise ValueError("literal_c0 may contain only depth-zero literal lookup")
            for stage in self.curriculum[1:]:
                rehearsal_depths = {profile.depth for profile in stage.profiles}
                if not {0, 1} <= rehearsal_depths:
                    raise ValueError(
                        f"{stage.name} must rehearse both literal C0 and C1 for A fairness"
                    )
        train_topologies = set(self.training_topologies)
        held_out_topologies = set(self.held_out_topologies)
        if not train_topologies or train_topologies & held_out_topologies:
            raise ValueError("training and held-out topology sets must be nonempty and disjoint")
        profile_topologies = {
            profile.topology
            for stage in self.curriculum
            for profile in stage.profiles
            if profile.topology != "leaf"
        }
        if profile_topologies != train_topologies:
            raise ValueError("training_topologies must exactly match non-leaf curriculum topologies")
        split_names = [split.name for split in self.evaluation_splits]
        if len(split_names) != len(set(split_names)):
            raise ValueError("evaluation split names must be unique")
        for split in self.evaluation_splits:
            split.validate()
            if split.category == "topology_extrapolation" and split.topology not in held_out_topologies:
                raise ValueError("topology extrapolation splits must use held-out topologies")
            if split.category != "topology_extrapolation" and split.topology not in train_topologies:
                raise ValueError("non-topology-extrapolation splits must use training topologies")
            if split.category == "in_distribution" and split.shape_partition != "train":
                raise ValueError("in-distribution splits must use the train shape partition")
            if split.category != "in_distribution" and split.shape_partition != "heldout":
                raise ValueError("extrapolation splits must use a held-out shape partition")
        categories = {split.category for split in self.evaluation_splits}
        if not {"in_distribution", "depth_extrapolation", "topology_extrapolation"} <= categories:
            raise ValueError("evaluation requires in-distribution, depth, and topology extrapolation splits")
        self.gate.validate()


def _profiles(raw: list[dict[str, object]]) -> tuple[TrainingProfile, ...]:
    return tuple(TrainingProfile(**item) for item in raw)


def _curriculum(raw: list[dict[str, object]]) -> tuple[CurriculumStage, ...]:
    return tuple(
        CurriculumStage(
            name=str(item["name"]),
            steps=int(item["steps"]),
            profiles=_profiles(item["profiles"]),
        )
        for item in raw
    )


def stage1_config_from_dict(raw: dict[str, object]) -> Stage1Config:
    nested = {
        "tasks",
        "eval_length_scales",
        "eval_seeds",
        "confirmation_training_seeds",
        "training_topologies",
        "held_out_topologies",
        "curriculum",
        "evaluation_splits",
        "gate",
        "data",
        "model_a",
        "model_d",
    }
    config = Stage1Config(
        **{key: value for key, value in raw.items() if key not in nested},
        tasks=tuple(raw.get("tasks", ("nested_expression",))),
        eval_length_scales=tuple(raw.get("eval_length_scales", (1, 2, 4))),
        eval_seeds=tuple(raw.get("eval_seeds", (11003, 22003, 33013))),
        confirmation_training_seeds=tuple(
            raw.get(
                "confirmation_training_seeds",
                (7301, 7307, 7321, 7331, 7333, 7349, 7351, 7369),
            )
        ),
        training_topologies=tuple(raw.get("training_topologies", ("skew", "balanced"))),
        held_out_topologies=tuple(raw.get("held_out_topologies", ("branched",))),
        curriculum=_curriculum(raw["curriculum"]) if "curriculum" in raw else _default_curriculum(),
        evaluation_splits=(
            tuple(EvaluationSplit(**item) for item in raw["evaluation_splits"])
            if "evaluation_splits" in raw
            else _default_splits()
        ),
        gate=GateConfig(**raw.get("gate", {})),
        data=DataConfig(**raw.get("data", {})),
        model_a=ModelConfig(**raw.get("model_a", {})),
        model_d=ModelConfig(**raw.get("model_d", {})),
    )
    config.validate()
    return config


def _validated_experiment_spec(
    config: Stage1Config | dict[str, object],
    allowed_variation_fields: set[str],
) -> dict[str, object]:
    validated = (
        config
        if isinstance(config, Stage1Config)
        else stage1_config_from_dict(config)
    )
    validated.validate()
    canonical = validated.to_dict()
    for name in allowed_variation_fields:
        canonical.pop(name, None)
    return canonical


def validated_experiment_spec(config: Stage1Config | dict[str, object]) -> dict[str, object]:
    return _validated_experiment_spec(
        config,
        EXPERIMENT_SPEC_ALLOWED_VARIATION_FIELDS,
    )


def validated_experiment_spec_digest(
    config: Stage1Config | dict[str, object],
) -> str:
    return stage1_config_digest(validated_experiment_spec(config))


def validated_experiment_compatibility_spec_digest(
    config: Stage1Config | dict[str, object],
) -> str:
    return stage1_config_digest(
        _validated_experiment_spec(
            config,
            EXPERIMENT_COMPATIBILITY_ALLOWED_VARIATION_FIELDS,
        )
    )


def load_stage1_config(path: str | Path) -> Stage1Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Stage 1 config root must be an object")
    return stage1_config_from_dict(raw)
