import json
import inspect
from typing import Type
from pydantic import BaseModel, Field, validator, ValidationError

"""
Configuration module for the viscous Cahn–Hilliard optimal control solver.

This file defines strongly typed data classes for the forward simulation and
optimization parameters.  The physical coefficients ``c1``, ``c2``, ``tau``, ``gamma``
and ``kappa`` govern the viscous Cahn–Hilliard system (1.2)–(1.4):
``c1`` and ``c2`` appear in the logarithmic potential f′(ϕ) = c₁ log((1+ϕ)/(1−ϕ)) − 2 c₂ ϕ;
``tau`` is the viscosity parameter multiplying ∂tϕ in equation (1.3);
``gamma`` is the relaxation time in the control channel γ ∂t w + w = u;
and ``kappa`` weights the gradient energy term −κ Δϕ in μ.

The optimisation weights ``b1``, ``b2``, ``b3`` and ``kappa_sparsity`` appear in the
cost functional J(ϕ,u) = (b1/2)∥ϕ−ϕ_Q∥² + (b2/2)∥ϕ(T)−ϕ_T∥² + (b3/2)∥u∥² +
κ‖u‖₁.  Large ``kappa_sparsity`` promotes sparsity in the optimal control by
penalising the L¹–norm of u, consistent with the sparsity condition u=0 whenever
|r| ≤ κ (Theorem 4.7).  Validators enforce sensible bounds on these
parameters (e.g. c2 > c1, u_max > u_min).
"""

# --- A helper function for clean user interaction ---
def get_yes_no_input(prompt: str) -> bool:
    """Gets a simple Yes/No confirmation from the user."""
    while True:
        response = input(f"{prompt} (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            return True
        if response in ['n', 'no']:
            return False
        print("Invalid input. Please enter 'y' or 'n'.")

def get_user_input_for_config(config_model: Type[BaseModel], title: str) -> BaseModel:
    """
    Interactively prompts the user for fields of a Pydantic model.
    Returns a fully constructed config_model instance.
    On invalid input, falls back to the field's default.
    """
    print("\n" + "="*len(title))
    print(title)
    print("="*len(title))

    user_params = {}

    # Iterate over __init__ signature so we keep order
    for param_name, param in inspect.signature(config_model).parameters.items():
        # Skip self/kwargs if present
        if param_name in ("self", "args", "kwargs"):
            continue

        # Pydantic v2: model_fields has metadata like description & default
        field_info = config_model.model_fields.get(param_name)
        default_val = param.default if param.default is not inspect._empty else (
            getattr(field_info, "default", None) if field_info else None
        )
        desc = (field_info.description if field_info and hasattr(field_info, "description") else "").strip()
        prompt = f"-> Enter '{param_name}'" + (f" ({desc})" if desc else "") + f" [default: {default_val}]: "

        while True:
            raw = input(prompt).strip()
            if raw == "":
                user_params[param_name] = default_val
                break

            # Try to cast to the annotated type if available
            ann = param.annotation
            try:
                if ann in (int, float, str):
                    user_params[param_name] = ann(raw)
                elif ann is bool:
                    user_params[param_name] = raw.lower() in ("y", "yes", "true", "1")
                else:
                    # If no clear annotation, keep string or try best-effort
                    user_params[param_name] = raw
                break
            except Exception as e:
                print(f"  [warn] Could not parse '{raw}' as {ann}. Error: {e}. Try again or press Enter for default.")

    # Always return a model instance; if something is invalid, fall back to defaults
    try:
        return config_model(**user_params)
    except Exception as e:
        print(f"[ERROR] Building {config_model.__name__} from inputs failed: {e}")
        print("Falling back to defaults.")
        return config_model()

# --- Configuration for the physical simulation ---
class ForwardSolverConfig(BaseModel):
    """Parameters controlling the Cahn-Hilliard forward simulation."""
    N: int = Field(128, gt=10, description="Number of spatial intervals")
    Lx: float = Field(1.0, gt=0, description="Domain length")
    T: float = Field(1.0, gt=0, description="Total simulation time")
    dt_initial: float = Field(1e-2, gt=0, description="Initial time step size")
    tau: float = Field(0.05, description="Viscosity  parameter for phi-equation")
    gamma: float = Field(10.0, gt=0, description="Relaxation parameter ")
    c1: float = Field(0.75, description="Flory-Huggins convex coefficient")
    c2: float = Field(1.0, description="Concave (quadratic) coefficient")
    # This is the gradient energy coefficient from the C-H equation
    kappa: float = Field(0.03**2, ge=0, description="Gradient energy coefficient")
    
    @validator('c2')
    def check_c2_greater_than_c1(cls, c2_val, values):
        c1_val = values.get('c1', 0)
        if c2_val <= c1_val:
            raise ValueError(f"c2 ({c2_val}) must be greater than c1 ({c1_val})")
        return c2_val


# --- Configuration for the optimization algorithm ---
class OptimizationConfig(BaseModel):
    """Parameters controlling the gradient descent optimization loop."""
    b1: float = Field(0.3, ge=0, description="Weight for space-time tracking cost")
    b2: float = Field(13.0, ge=0, description="Weight for terminal cost")
    b3: float = Field(0.0019, ge=0, description="Weight for control energy cost")
    # This kappa is for the L1 sparsity term in the cost function
    kappa_sparsity: float = Field(0.00009, ge=0, description="Sparsity weight for L1 term")
    alpha_max: float = Field(100.0, gt=0, description="Initial step size for line search")
    max_iter: int = Field(1000, gt=10, description="Max number of gradient descent iterations")
    u_min: float = Field(-1.0, description="Lower bound for the control")
    u_max: float = Field(1.0, description="Upper bound for the control")

    @validator('u_max')
    def u_max_must_be_greater_than_u_min(cls, u_max_val, values):
        if 'u_min' in values and u_max_val <= values['u_min']:
            raise ValueError("u_max must be strictly greater than u_min.")
        return u_max_val




# --- NEW: A main container for all parameters ---
class SimulationParameters(BaseModel):
    """A single container to hold all simulation configurations and metadata."""
    forward_solver: ForwardSolverConfig = Field(default_factory=ForwardSolverConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    last_run_iterations: int = Field(0, description="Number of iterations from the last run.")

# --- NEW: Function to save parameters to a file ---
def save_params(
    fwd_config: ForwardSolverConfig,
    opt_config: OptimizationConfig,
    iteration_count: int,
    filepath: str = "last_run_config.json"
):
    """Saves the used configurations and final iteration count to a JSON file."""
    params = SimulationParameters(
        forward_solver=fwd_config,
        optimization=opt_config,
        last_run_iterations=iteration_count
    )
    try:
        with open(filepath, "w") as f:
            f.write(params.model_dump_json(indent=4))
        print(f"\n✅ Configuration saved to '{filepath}' for your next session.")
    except IOError as e:
        print(f"\n[Warning] Could not save configuration file: {e}")

# --- NEW: Function to load parameters from a file ---
def load_params(filepath: str = "last_run_config.json") -> SimulationParameters:
    """Loads simulation parameters from a JSON file, or returns defaults."""
    try:
        with open(filepath, "r") as f:
            data = json.load(f)
            print(f"✅ Loaded previous configuration from '{filepath}'.")
            return SimulationParameters(**data)
    except (FileNotFoundError, ValidationError, json.JSONDecodeError):
        print("No valid previous configuration found. Using default parameters.")
        return SimulationParameters()

# --- MODIFIED: Update the interactive prompter ---
# In config.py



# In config.py

def get_user_input_for_config(
    config_model: Type[BaseModel],
    title: str,
    previous_instance: BaseModel | None = None  # Changed parameter name
) -> BaseModel:
    """
    Interactively prompts for parameters. It displays values from a previous run
    in a table for reference, but always uses the original hardcoded defaults
    in the input prompts.
    """
    print("\n" + "="*60)
    print(f"--- {title} ---")

    # --- NEW: Display previous parameters in a table if they exist ---
    if previous_instance:
        print("For your reference, here are the parameters from the last run:")
        print("."*50)
        for param_name in previous_instance.model_fields:
            value = getattr(previous_instance, param_name)
            print(f"  {param_name:<15}: {value}")
        print("."*50)

    print("Please provide new parameters below.")
    print("Press Enter to accept the original default value shown in [brackets].")
    print("="*60)

    user_params = {}
    model_fields = config_model.model_fields

    # --- Step 1: Initial pass to gather all parameters ---
    # This loop now correctly uses the original defaults for the prompts
    for param_name, param in inspect.signature(config_model).parameters.items():
        field_info = model_fields.get(param_name)
        if not field_info:
            continue

        # *** MODIFICATION: Always use the original default from the class definition ***
        original_default = param.default
        prompt = f"-> Enter '{param_name}' ({field_info.description}) [default: {original_default}]: "

        while True: # This inner loop handles basic type errors
            user_input = input(prompt).strip()
            if not user_input:
                user_params[param_name] = original_default
                break
            else:
                try:
                    user_params[param_name] = field_info.annotation(user_input)
                    break
                except (ValueError, TypeError):
                    print(f"  [Error] Invalid format. Please enter a value of type '{field_info.annotation.__name__}'.")

    # --- Step 2: Validation loop (remains the same) ---
    while True:
        try:
            validated_config = config_model(**user_params)
            print("\n✓ Configuration accepted and validated.")
            return validated_config

        except ValidationError as e:
            print("\n" + "!"*60)
            print("🚨 PARAMETER ERROR: Please correct the following value(s):")
            invalid_fields = {err['loc'][0] for err in e.errors()}
            for error in e.errors():
                field = error['loc'][0]
                message = error['msg']
                print(f"  - {field}: {message}")
            print("!"*60)

            # Re-prompt *only* for the fields that had an error
            for param_name in invalid_fields:
                field_info = model_fields[param_name]
                original_default = inspect.signature(config_model).parameters[param_name].default
                prompt = f"-> (Correction) Enter '{param_name}' ({field_info.description}) [default: {original_default}]: "
                
                while True:
                    user_input = input(prompt).strip()
                    if not user_input:
                        user_params[param_name] = original_default
                        break
                    else:
                        try:
                            user_params[param_name] = field_info.annotation(user_input)
                            break
                        except (ValueError, TypeError):
                            print(f"  [Error] Invalid format. Please enter a value of type '{field_info.annotation.__name__}'.")