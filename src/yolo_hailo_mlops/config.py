"""Configuration loading, validation, and multi-tier precedence resolution."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import yaml
except ImportError:
    yaml = None

from yolo_hailo_mlops.exceptions import ConfigurationError
from yolo_hailo_mlops.state import ExecutionMode


@dataclass
class ProjectConfig:
    name: str = "yolo11-hailo"
    seed: int = 42


@dataclass
class ModelConfig:
    source: str = "yolo11n.pt"
    variant: str = "n"
    task: str = "detect"


@dataclass
class CalibrationDatasetConfig:
    source: str = "train"
    count: int = 200
    seed: int = 42
    custom_dir: Optional[str] = None


@dataclass
class DatasetConfig:
    yaml: str = "data/dataset.yaml"
    calibration: CalibrationDatasetConfig = field(default_factory=CalibrationDatasetConfig)


@dataclass
class TrainingConfig:
    epochs: int = 100
    imgsz: int = 640
    batch: int = 16
    device: str = "auto"
    optimizer: str = "auto"
    lr0: Optional[float] = None
    lrf: Optional[float] = None
    patience: int = 50
    deterministic: bool = True


@dataclass
class ExportConfig:
    imgsz: int = 640
    batch: int = 1
    dynamic: bool = False
    precision: str = "fp32"
    simplify: bool = True
    opset: Optional[int] = None
    nms: str = "auto"


@dataclass
class HailoConfig:
    target: str = "hailo8l"
    sdk_mode: str = "auto"
    require_hardware: bool = False
    har_path: Optional[str] = None
    hef_path: Optional[str] = None
    compiler_optimization_level: int = 0
    calib_batch_size: int = 1
    start_node_names: Optional[List[str]] = None
    end_node_names: Optional[List[str]] = None


@dataclass
class QuantizationConfig:
    mode: str = "int8_ptq"
    optimisation: bool = True


@dataclass
class AccuracyGateConfig:
    map50_max_drop: float = 0.02
    map5095_max_drop: float = 0.02


@dataclass
class PerformanceConfig:
    warmup_iterations: int = 20
    measurement_iterations: int = 100


@dataclass
class ValidationConfig:
    enabled: bool = True
    accuracy: AccuracyGateConfig = field(default_factory=AccuracyGateConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)


@dataclass
class ArtifactsConfig:
    root: str = "./artifacts"


@dataclass
class AppConfig:
    """Master application configuration tree."""
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    hailo: HailoConfig = field(default_factory=HailoConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    # Runtime operational controls
    run_id: Optional[str] = None
    execution_mode: ExecutionMode = ExecutionMode.AUTO
    dry_run: bool = False
    force: bool = False
    resume: bool = False
    verbose: bool = False
    config_file: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        d = asdict(self)
        d["execution_mode"] = self.execution_mode.value
        return d

    def validate(self) -> None:
        """Validate logical invariants across configuration parameters."""
        # Project validation
        if not self.project.name.strip():
            raise ConfigurationError("project.name cannot be empty", code="E-CFG-002")
        if self.project.seed < 0:
            raise ConfigurationError("project.seed must be non-negative", code="E-CFG-003")

        # Model validation
        valid_variants = {"n", "s", "m", "l", "x", "custom"}
        if self.model.variant not in valid_variants:
            raise ConfigurationError(
                f"Invalid model variant '{self.model.variant}'. Expected one of {valid_variants}",
                code="E-CFG-004",
            )
        if self.model.task != "detect":
            raise ConfigurationError(
                f"Unsupported task '{self.model.task}'. Only 'detect' is supported.",
                code="E-CFG-005",
            )

        # Dataset & calibration validation
        if self.dataset.calibration.source not in {"train", "custom"}:
            raise ConfigurationError(
                f"Invalid calibration source '{self.dataset.calibration.source}'. Expected 'train' or 'custom'.",
                code="E-CFG-006",
            )
        if self.dataset.calibration.count < 1:
            raise ConfigurationError(
                f"Calibration count must be >= 1, got {self.dataset.calibration.count}",
                code="E-CFG-007",
            )
        if self.dataset.calibration.source == "custom" and not self.dataset.calibration.custom_dir:
            raise ConfigurationError(
                "dataset.calibration.custom_dir must be specified when source is 'custom'",
                code="E-CFG-008",
            )

        # Training validation
        if self.training.epochs < 1:
            raise ConfigurationError("training.epochs must be >= 1", code="E-CFG-009")
        if self.training.imgsz < 32 or self.training.imgsz % 32 != 0:
            raise ConfigurationError("training.imgsz must be a multiple of 32 (e.g. 640)", code="E-CFG-010")
        if self.training.batch < 1:
            raise ConfigurationError("training.batch must be >= 1", code="E-CFG-011")

        # Export validation
        if self.export.batch != 1:
            raise ConfigurationError("export.batch must be static 1 for Hailo-8L target", code="E-CFG-012")
        if self.export.dynamic:
            raise ConfigurationError("export.dynamic must be false for canonical Hailo-8L compilation", code="E-CFG-013")
        if self.export.precision not in {"fp32", "fp16"}:
            raise ConfigurationError("export.precision must be 'fp32' or 'fp16'", code="E-CFG-014")

        # Hailo validation
        if self.hailo.target != "hailo8l":
            raise ConfigurationError(
                f"Unsupported target '{self.hailo.target}'. Target must be 'hailo8l' for Raspberry Pi 5 AI HAT+.",
                code="E-CFG-015",
            )
        if self.hailo.sdk_mode not in {"auto", "sdk", "cli"}:
            raise ConfigurationError(
                f"Invalid hailo.sdk_mode '{self.hailo.sdk_mode}'. Expected 'auto', 'sdk', or 'cli'.",
                code="E-CFG-016",
            )

        # Accuracy gates
        if not (0.0 <= self.validation.accuracy.map50_max_drop <= 1.0):
            raise ConfigurationError("map50_max_drop must be between 0.0 and 1.0", code="E-CFG-017")
        if not (0.0 <= self.validation.accuracy.map5095_max_drop <= 1.0):
            raise ConfigurationError("map5095_max_drop must be between 0.0 and 1.0", code="E-CFG-018")


def load_config(
    config_path: Optional[str | Path] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> AppConfig:
    """Load configuration applying strict precedence: CLI > Env > YAML > Defaults."""
    raw: Dict[str, Any] = {}

    # 1. Determine configuration file path
    target_config = config_path or os.getenv("YOLO_HAILO_CONFIG", "config/config.yaml")
    config_file_path = Path(target_config)

    if config_file_path.exists():
        if yaml is not None:
            try:
                with open(config_file_path, "r", encoding="utf-8") as f:
                    loaded = yaml.safe_load(f)
                    if isinstance(loaded, dict):
                        raw = loaded
            except Exception as e:
                raise ConfigurationError(
                    f"Failed to parse YAML configuration file '{config_file_path}': {e}",
                    code="E-CFG-001",
                    artifact=str(config_file_path),
                ) from e

    # 2. Extract nested dictionaries
    proj_d = raw.get("project", {})
    model_d = raw.get("model", {})
    ds_d = raw.get("dataset", {})
    calib_d = ds_d.get("calibration", {})
    train_d = raw.get("training", {})
    exp_d = raw.get("export", {})
    hailo_d = raw.get("hailo", {})
    quant_d = raw.get("quantization") or raw.get("quantisation") or {}
    val_d = raw.get("validation", {})
    acc_d = val_d.get("accuracy", {})
    perf_d = val_d.get("performance", {})
    art_d = raw.get("artifacts") or raw.get("artefacts") or {}

    # 3. Apply Environment Variable overrides
    def get_env_val(key: str, default: Any, caster: type = str) -> Any:
        val = os.getenv(key)
        if val is None:
            return default
        if caster == bool:
            return val.lower() in ("true", "1", "yes")
        try:
            return caster(val)
        except (ValueError, TypeError):
            return default

    # Build typed sub-configs
    project_cfg = ProjectConfig(
        name=get_env_val("YOLO_HAILO_PROJECT_NAME", proj_d.get("name", "yolo11-hailo")),
        seed=get_env_val("YOLO_HAILO_SEED", proj_d.get("seed", 42), int),
    )

    model_cfg = ModelConfig(
        source=get_env_val("YOLO_HAILO_MODEL_SOURCE", model_d.get("source", "yolo11n.pt")),
        variant=get_env_val("YOLO_HAILO_MODEL_VARIANT", model_d.get("variant", "n")),
        task=get_env_val("YOLO_HAILO_MODEL_TASK", model_d.get("task", "detect")),
    )

    calib_cfg = CalibrationDatasetConfig(
        source=get_env_val("YOLO_HAILO_CALIB_SOURCE", calib_d.get("source", "train")),
        count=get_env_val("YOLO_HAILO_CALIB_COUNT", calib_d.get("count", 200), int),
        seed=get_env_val("YOLO_HAILO_CALIB_SEED", calib_d.get("seed", 42), int),
        custom_dir=get_env_val("YOLO_HAILO_CALIB_CUSTOM_DIR", calib_d.get("custom_dir")),
    )

    dataset_cfg = DatasetConfig(
        yaml=get_env_val("YOLO_HAILO_DATASET_YAML", ds_d.get("yaml", "data/dataset.yaml")),
        calibration=calib_cfg,
    )

    train_cfg = TrainingConfig(
        epochs=get_env_val("YOLO_HAILO_EPOCHS", train_d.get("epochs", 100), int),
        imgsz=get_env_val("YOLO_HAILO_IMGSZ", train_d.get("imgsz", 640), int),
        batch=get_env_val("YOLO_HAILO_BATCH", train_d.get("batch", 16), int),
        device=get_env_val("YOLO_HAILO_DEVICE", train_d.get("device", "auto")),
        optimizer=get_env_val("YOLO_HAILO_OPTIMIZER", train_d.get("optimizer", "auto")),
        lr0=train_d.get("lr0"),
        lrf=train_d.get("lrf"),
        patience=get_env_val("YOLO_HAILO_PATIENCE", train_d.get("patience", 50), int),
        deterministic=get_env_val("YOLO_HAILO_DETERMINISTIC", train_d.get("deterministic", True), bool),
    )

    exp_cfg = ExportConfig(
        imgsz=get_env_val("YOLO_HAILO_EXPORT_IMGSZ", exp_d.get("imgsz", 640), int),
        batch=1,
        dynamic=False,
        precision=get_env_val("YOLO_HAILO_PRECISION", exp_d.get("precision", "fp32")),
        simplify=get_env_val("YOLO_HAILO_SIMPLIFY", exp_d.get("simplify", True), bool),
        opset=exp_d.get("opset"),
        nms=get_env_val("YOLO_HAILO_NMS", exp_d.get("nms", "auto")),
    )

    hailo_cfg = HailoConfig(
        target=get_env_val("YOLO_HAILO_TARGET", hailo_d.get("target", "hailo8l")),
        sdk_mode=get_env_val("YOLO_HAILO_SDK_MODE", hailo_d.get("sdk_mode", "auto")),
        require_hardware=get_env_val("YOLO_HAILO_REQUIRE_HARDWARE", hailo_d.get("require_hardware", False), bool),
        har_path=get_env_val("YOLO_HAILO_HAR_PATH", hailo_d.get("har_path")),
        hef_path=get_env_val("YOLO_HAILO_HEF_PATH", hailo_d.get("hef_path")),
        compiler_optimization_level=hailo_d.get("compiler_optimization_level", 0),
        calib_batch_size=hailo_d.get("calib_batch_size", 1),
    )

    quant_cfg = QuantizationConfig(
        mode=get_env_val("YOLO_HAILO_QUANT_MODE", quant_d.get("mode", "int8_ptq")),
        optimisation=get_env_val("YOLO_HAILO_QUANT_OPT", quant_d.get("optimisation", True), bool),
    )

    acc_cfg = AccuracyGateConfig(
        map50_max_drop=get_env_val("YOLO_HAILO_MAP50_MAX_DROP", acc_d.get("map50_max_drop", 0.02), float),
        map5095_max_drop=get_env_val("YOLO_HAILO_MAP5095_MAX_DROP", acc_d.get("map5095_max_drop", 0.02), float),
    )

    perf_cfg = PerformanceConfig(
        warmup_iterations=get_env_val("YOLO_HAILO_WARMUP", perf_d.get("warmup_iterations", 20), int),
        measurement_iterations=get_env_val("YOLO_HAILO_MEASURE", perf_d.get("measurement_iterations", 100), int),
    )

    val_cfg = ValidationConfig(
        enabled=get_env_val("YOLO_HAILO_VAL_ENABLED", val_d.get("enabled", True), bool),
        accuracy=acc_cfg,
        performance=perf_cfg,
    )

    art_cfg = ArtifactsConfig(
        root=get_env_val("YOLO_HAILO_ARTIFACTS_ROOT", art_d.get("root", "./artifacts")),
    )

    # Execution controls
    exec_mode_str = get_env_val("YOLO_HAILO_EXECUTION_MODE", "auto")
    try:
        exec_mode = ExecutionMode(exec_mode_str.lower())
    except ValueError:
        exec_mode = ExecutionMode.AUTO

    app_cfg = AppConfig(
        project=project_cfg,
        model=model_cfg,
        dataset=dataset_cfg,
        training=train_cfg,
        export=exp_cfg,
        hailo=hailo_cfg,
        quantization=quant_cfg,
        validation=val_cfg,
        artifacts=art_cfg,
        run_id=get_env_val("YOLO_HAILO_RUN_ID", None),
        execution_mode=exec_mode,
        dry_run=get_env_val("YOLO_HAILO_DRY_RUN", False, bool),
        force=get_env_val("YOLO_HAILO_FORCE", False, bool),
        resume=get_env_val("YOLO_HAILO_RESUME", False, bool),
        verbose=get_env_val("YOLO_HAILO_VERBOSE", False, bool),
        config_file=str(config_file_path) if config_file_path.exists() else None,
    )

    # 4. Apply CLI argument overrides (Highest Precedence)
    if cli_overrides:
        if "model" in cli_overrides and cli_overrides["model"]:
            app_cfg.model.source = str(cli_overrides["model"])
        if "dataset" in cli_overrides and cli_overrides["dataset"]:
            app_cfg.dataset.yaml = str(cli_overrides["dataset"])
        if "device" in cli_overrides and cli_overrides["device"]:
            app_cfg.training.device = str(cli_overrides["device"])
        if "epochs" in cli_overrides and cli_overrides["epochs"] is not None:
            app_cfg.training.epochs = int(cli_overrides["epochs"])
        if "imgsz" in cli_overrides and cli_overrides["imgsz"] is not None:
            app_cfg.training.imgsz = int(cli_overrides["imgsz"])
            app_cfg.export.imgsz = int(cli_overrides["imgsz"])
        if "batch" in cli_overrides and cli_overrides["batch"] is not None:
            app_cfg.training.batch = int(cli_overrides["batch"])
        if "seed" in cli_overrides and cli_overrides["seed"] is not None:
            app_cfg.project.seed = int(cli_overrides["seed"])
            app_cfg.dataset.calibration.seed = int(cli_overrides["seed"])
        if "run_id" in cli_overrides and cli_overrides["run_id"]:
            app_cfg.run_id = str(cli_overrides["run_id"])
        if "force" in cli_overrides and cli_overrides["force"] is not None:
            app_cfg.force = bool(cli_overrides["force"])
        if "resume" in cli_overrides and cli_overrides["resume"] is not None:
            app_cfg.resume = bool(cli_overrides["resume"])
        if "verbose" in cli_overrides and cli_overrides["verbose"] is not None:
            app_cfg.verbose = bool(cli_overrides["verbose"])
        if "dry_run" in cli_overrides and cli_overrides["dry_run"] is not None:
            app_cfg.dry_run = bool(cli_overrides["dry_run"])
        if "execution_mode" in cli_overrides and cli_overrides["execution_mode"]:
            try:
                app_cfg.execution_mode = ExecutionMode(cli_overrides["execution_mode"].lower())
            except ValueError:
                pass
        if "require_hardware" in cli_overrides and cli_overrides["require_hardware"] is not None:
            app_cfg.hailo.require_hardware = bool(cli_overrides["require_hardware"])

    app_cfg.validate()
    return app_cfg
