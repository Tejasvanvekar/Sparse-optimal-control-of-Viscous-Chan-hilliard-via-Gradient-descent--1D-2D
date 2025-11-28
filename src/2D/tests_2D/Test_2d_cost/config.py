
"""
Unified configuration & interactive prompts for the 2D Cahn–Hilliard optimizer.

This module defines typed configuration objects for the forward solver
(:class:`ForwardSolverConfig`) and optimization loop (:class:`OptimizationConfig`)
using the Pydantic library.  These models provide validation and default
values for all simulation parameters and cost weights.  The code is
compatible with both Pydantic v1 and v2 by using helper functions
(:func:`_model_dump`, :func:`_model_dump_json`, :func:`_get_model_fields`) to
abstract away API differences.

In addition to the data models, the module supplies I/O helpers to save
and load configurations to JSON, and interactive functions to prompt a
user for parameter values.  The interactive prompts use simple
``input()`` calls and show previous values to make iterative tuning
easier.  When running in non‑interactive mode, these prompts can be
skipped by setting flags in the calling script.
"""
import json
from typing import Type, Optional, Any, Dict

from pydantic import BaseModel, Field, validator, ValidationError


# ---------- Pydantic v1/v2 compatibility helpers ----------
def _model_dump(obj: BaseModel) -> Dict[str, Any]:
    """Return a dict of fields for both Pydantic v1 and v2 models."""
    try:
        return obj.model_dump()  # Pydantic v2
    except Exception:
        return obj.dict()        # Pydantic v1


def _model_dump_json(obj: BaseModel, indent: int = 4) -> str:
    """Return JSON for both Pydantic v1 and v2 models."""
    try:
        return obj.model_dump_json(indent=indent)  # v2
    except Exception:
        return json.dumps(obj.dict(), indent=indent)  # v1


def _get_model_fields(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """Return a mapping of field_name -> field_info for both v1 and v2."""
    try:
        return model_cls.model_fields  # v2
    except Exception:
        return model_cls.__fields__    # v1


def _field_default_and_desc(field_info: Any) -> (Any, str):
    """Extract default and description from a field (v1/v2 safe)."""
    # Default
    default = getattr(field_info, "default", None)
    if default is None and hasattr(field_info, "default_factory"):
        try:
            default = field_info.default_factory()  # type: ignore[attr-defined]
        except Exception:
            default = None

    # Description
    desc = getattr(field_info, "description", None)
    if desc is None and hasattr(field_info, "field_info"):
        # v1 keeps description in field.field_info.description
        desc = getattr(field_info.field_info, "description", None)

    return default, (desc or "")


# ---------- Simple yes/no helper ----------
def get_yes_no_input(prompt: str) -> bool:
    """Gets a simple Yes/No confirmation from the user (no TTY gating here)."""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ("y", "yes"):
            return True
        if response in ("n", "no"):
            return False
        print("Invalid input. Please enter 'y' or 'n'.")


# ---------- Simulation configs ----------
class ForwardSolverConfig(BaseModel):
    """
    Parameters controlling the 2D Cahn–Hilliard forward simulation.

    The forward solver discretizes the Cahn–Hilliard equation on a 2D rectangular
    domain with Neumann boundary conditions.  The fields defined here specify
    the spatial resolution (``Nx`` × ``Ny``), physical dimensions (``Lx`` × ``Ly``),
    total simulation time ``T``, initial time‑step size ``dt_initial``, and
    several physical parameters:

    - ``tau``: inertial relaxation time for the phase equation;
    - ``gamma``: relaxation parameter for the auxiliary variable ``w``;
    - ``c1`` and ``c2``: coefficients in the Flory–Huggins free energy
      splitting (convex and concave parts, respectively);
    - ``kappa``: gradient energy coefficient controlling interfacial width.

    These defaults provide a reasonable starting point for experimentation,
    but users are encouraged to adjust them to explore different regimes of
    phase separation dynamics.
    """
    Nx: int = Field(128, gt=10, description="Number of spatial intervals in x")
    Ny: int = Field(128, gt=10, description="Number of spatial intervals in y")
    Lx: float = Field(1.0, gt=0, description="Domain length in x")
    Ly: float = Field(1.0, gt=0, description="Domain length in y")
    T: float = Field(1.0, gt=0, description="Total simulation time")
    dt_initial: float = Field(1e-2, gt=0, description="Initial time step size")
    tau: float = Field(0.05, description="viscosity parameter for phi-equation")
    gamma: float = Field(10.0, gt=0, description="Relaxation parameter ")
    c1: float = Field(0.75, description="Flory–Huggins convex coefficient")
    c2: float = Field(1.0, description="Concave (quadratic) coefficient")
    kappa: float = Field(0.01**2, ge=0, description="Gradient energy coefficient")

    @validator("c2")
    def check_c2_greater_than_c1(cls, c2_val, values):
        c1_val = values.get("c1", 0.0)
        if c2_val <= c1_val:
            raise ValueError(f"c2 ({c2_val}) must be greater than c1 ({c1_val})")
        return c2_val


class OptimizationConfig(BaseModel):
    """
    Parameters controlling the gradient‑descent optimization loop.

    This model encapsulates all weights and constraints used in the
    gradient‑descent procedure that seeks an optimal control ``u``.  The
    objective function consists of multiple contributions: a space–time
    tracking cost weighted by ``b1``; a terminal cost weighted by ``b2``;
    a quadratic control energy weighted by ``b3``; and an optional L1
    sparsity term weighted by ``kappa_sparsity``.  The step size for the
    backtracking line search is bounded by ``alpha_max``, and the maximum
    number of iterations allowed is ``max_iter``.  The admissible control
    values are restricted to the interval [``u_min``, ``u_max``].
    """
    b1: float = Field(5.0, ge=0, description="Weight for space-time tracking cost")
    b2: float = Field(10.0, ge=0, description="Weight for terminal cost")
    b3: float = Field(0.0001, ge=0, description="Weight for control energy cost")
    kappa_sparsity: float = Field(1e-4, ge=0, description="Sparsity weight for L1 term")
    alpha_max: float = Field(50.0, gt=0, description="Initial step size for line search")
    max_iter: int = Field(500, gt=10, description="Max number of gradient descent iterations")
    u_min: float = Field(-1.0, description="Lower bound for the control")
    u_max: float = Field(1.0, description="Upper bound for the control")

    @validator("u_max")
    def u_max_must_be_greater_than_u_min(cls, u_max_val, values):
        if "u_min" in values and u_max_val <= values["u_min"]:
            raise ValueError("u_max must be strictly greater than u_min.")
        return u_max_val


class SimulationParameters(BaseModel):
    """Container to hold both solver & optimization configs plus metadata."""
    forward_solver: ForwardSolverConfig = Field(default_factory=ForwardSolverConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    last_run_iterations: int = Field(0, description="Number of iterations from the last run.")


# ---------- Save/load helpers ----------
def save_params(
    fwd_config: ForwardSolverConfig,
    opt_config: OptimizationConfig,
    iteration_count: int,
    filepath: str = "last_run_config_2d.json",
) -> None:
    """Saves configs and iteration count to JSON (v1/v2 compatible)."""
    params = SimulationParameters(
        forward_solver=fwd_config,
        optimization=opt_config,
        last_run_iterations=iteration_count,
    )
    try:
        with open(filepath, "w") as f:
            f.write(_model_dump_json(params, indent=4))
        print(f"\n✅ Configuration saved to '{filepath}' for your next session.")
    except IOError as e:
        print(f"\n[Warning] Could not save configuration file: {e}")


def load_params(filepath: str = "last_run_config_2d.json") -> SimulationParameters:
    """Loads configs from JSON or returns default parameters."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
        print(f"✅ Loaded previous configuration from '{filepath}'.")
        return SimulationParameters(**data)
    except (FileNotFoundError, ValidationError, json.JSONDecodeError):
        print("No valid previous configuration found. Using default parameters.")
        return SimulationParameters()


# ---------- Interactive config editor ----------
def get_user_input_for_config(
    config_model: Type[BaseModel],
    title: str,
    previous_instance: Optional[BaseModel] = None,  # Py3.8-friendly
) -> BaseModel:
    """
    Interactively prompt the user for a configuration model.
    - Shows previous values (if any).
    - Validates with Pydantic; re-prompts only erroneous fields.
    """
    print("\n" + "=" * 60)
    print(f"--- {title} ---")

    if previous_instance is not None:
        print("For your reference, here are the parameters from the last run:")
        print("." * 50)
        for param_name, value in _model_dump(previous_instance).items():
            print(f"  {param_name:<15}: {value}")
        print("." * 50)

    print("Please provide new parameters below.")
    print("Press Enter to accept the original default value shown in [brackets].")
    print("=" * 60)

    user_params: Dict[str, Any] = {}
    model_fields = _get_model_fields(config_model)

    # First pass: prompt all fields
    for param_name, field_info in model_fields.items():
        default, desc = _field_default_and_desc(field_info)
        prompt = f"-> Enter '{param_name}' ({desc}) [default: {default}]: "

        user_input = input(prompt).strip()
        if user_input == "":
            user_params[param_name] = default
        else:
            # Store raw string; Pydantic will coerce types
            user_params[param_name] = user_input

    # Validation/re-prompt loop
    while True:
        try:
            validated = config_model(**user_params)
            print("\n✓ Configuration accepted and validated.")
            return validated
        except ValidationError as e:
            print("\n" + "!" * 60)
            print("🚨 PARAMETER ERROR: Please correct the following value(s):")
            invalid_fields = {err["loc"][0] for err in e.errors() if err.get("loc")}
            for error in e.errors():
                field = error["loc"][0]
                message = error["msg"]
                print(f"  - {field}: {message}")
            print("!" * 60)

            for param_name in invalid_fields:
                field_info = model_fields[param_name]
                default, desc = _field_default_and_desc(field_info)
                prompt = f"-> (Correction) Enter '{param_name}' ({desc}) [default: {default}]: "
                user_input = input(prompt).strip()
                if user_input == "":
                    user_params[param_name] = default
                else:
                    user_params[param_name] = user_input
