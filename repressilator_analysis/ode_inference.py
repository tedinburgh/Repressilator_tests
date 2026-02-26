"""
ODE model and parameter inference for the Repressilator system.

This module implements the Repressilator ODE model and uses PINTS
for Bayesian parameter inference.
"""

import numpy as np
import pints
from scipy.integrate import odeint
from typing import List, Dict


class RepressilatorModel(pints.ForwardModel):
    """
    ODE model for the Repressilator genetic circuit.

    The Repressilator consists of three repressors (LacI, TetR, CI) that
    form a cyclic negative feedback loop. Each repressor inhibits the
    transcription of the next repressor in the cycle.

    State variables:
    - m1, m2, m3: mRNA concentrations for repressors 1, 2, 3
    - p1, p2, p3: Protein concentrations for repressors 1, 2, 3

    Parameters:
    - alpha: Transcription rate
    - alpha0: Basal transcription rate
    - hill: Hill coefficient (cooperativity)
    - K_m: Repression threshold concentration in Hill term
    - T_e: Expression timescale
    - mrna_half_life: mRNA half-life (minutes)
    - p_half_life: Protein half-life (minutes)
    - initial_c_p: Initial cytosolic protein
    - initial_n_p_1: Initial nuclear protein 1
    - initial_n_p_2: Initial nuclear protein 2
    - initial_c_m: Initial cytosolic mRNA
    - initial_n_m_1: Initial nuclear mRNA 1
    - initial_n_m_2: Initial nuclear mRNA 2
    """
    _param_names=[
        "alpha",
        "alpha0",
        "hill",
        "K_m",
        "T_e",
        "mrna_half_life",
        "p_half_life",
        "initial_c_p",
        "initial_n_p_1",
        "initial_n_p_2",
        "initial_c_m",
        "initial_n_m_1",
        "initial_n_m_2",
    ]
    def __init__(self, times: np.ndarray):
        """
        Initialize the Repressilator model.

        Args:
            times: Array of time points for simulation
        """
        self.times = times

    def n_parameters(self) -> int:
        """Return the number of model parameters."""
        return 13

    def n_outputs(self) -> int:
        """Return the number of observable outputs."""
        # We observe 2 proteins (nuclear and cytoplasmic)
        return 2

    def simulate(self, parameters: List[float], times: np.ndarray) -> np.ndarray:
        """
        Simulate the Repressilator ODE system.

        Args:
            parameters: Model parameters in RepressilatorModel._param_names order
            times: Time points for simulation

        Returns:
            Array of shape (n_times, n_outputs) with protein concentrations
        """
        if len(parameters) != self.n_parameters():
            raise ValueError(
                f"Expected {self.n_parameters()} parameters, got {len(parameters)}"
            )
        (
            alpha,
            alpha0,
            hill,
            K_m,
            T_e,
            mrna_half_life,
            p_half_life,
            initial_c_p,
            initial_n_p_1,
            initial_n_p_2,
            initial_c_m,
            initial_n_m_1,
            initial_n_m_2,
        ) = parameters

        beta = p_half_life / mrna_half_life

        # Derived basal term used in nondimensional repressilator forms.
        alpha_0 = alpha * alpha0

        # Convert half-lives to degradation rates: degradation_rate = ln(2) / half_life
        gamma_m = np.log(2) / mrna_half_life
        gamma_p = np.log(2) / p_half_life

        # Initial conditions from parameter vector.
        # State order is [m1, m2, m3, p1, p2, p3].
        y0 = [
            initial_n_m_1,
            initial_c_m,
            initial_n_m_2,
            initial_n_p_1,
            initial_c_p,
            initial_n_p_2,
        ]

        def repressilator_odes(y, t):
            """ODE system for the Repressilator."""
            m1, m2, m3, p1, p2, p3 = y

            # Hill function for repression
            def hill_repression(repressor_conc):
                scaled = repressor_conc / K_m
                return alpha / (1 + (scaled ** hill)) + alpha_0

            # mRNA dynamics
            dm1_dt = (hill_repression(p3) - gamma_m * m1) / T_e
            dm2_dt = (hill_repression(p1) - gamma_m * m2) / T_e
            dm3_dt = (hill_repression(p2) - gamma_m * m3) / T_e

            # Protein dynamics
            dp1_dt = beta * m1 - gamma_p * p1
            dp2_dt = beta * m2 - gamma_p * p2
            dp3_dt = beta * m3 - gamma_p * p3

            return [dm1_dt, dm2_dt, dm3_dt, dp1_dt, dp2_dt, dp3_dt]

        # Solve ODE system
        solution = odeint(repressilator_odes, y0, times)

        # Extract observable proteins (p1 and p2 as nuclear and cytoplasmic)
        # Note: p3 is the unobserved protein without fluorescence
        output = solution[:, [3, 4]]  # [p1, p2]

        return output


def infer_parameters(
    times: np.ndarray,
    observations: np.ndarray,
    method: str = 'cmaes',
) -> np.ndarray:
    """
    Infer Repressilator parameters using optimization.

    Args:
        times: Time points (in minutes)
        observations: Observed protein concentrations of shape (n_times, 2)
        method: Optimization method ('cmaes' or 'xnes')

    Returns:
        Best-fit parameters in RepressilatorModel._param_names order
    """
    # Create model
    model = RepressilatorModel(times)

    # Define parameter bounds
    bounds = [
        (0, 1000),    # alpha
        (0, 10),      # alpha0
        (1, 5),       # hill
        (1, 1000),    # K_m
        (1, 100),     # T_e
        (0.693, 69.3),  # mrna_half_life
        (6.93, 693),    # p_half_life
        (0, 1000),    # initial_c_p
        (0, 1000),    # initial_n_p_1
        (0, 1000),    # initial_n_p_2
        (0, 1000),    # initial_c_m
        (0, 1000),    # initial_n_m_1
        (0, 1000),    # initial_n_m_2
    ]

    # PINTS optimization problem (sum of squared residuals over both outputs).
    problem = pints.MultiOutputProblem(model, times, observations)
    cost_function = pints.SumOfSquaresError(problem)

    lower = [x[0] for x in bounds]
    upper = [x[1] for x in bounds]
    boundaries = pints.RectangularBoundaries(lower, upper)

    # Initial guess
    x0 = np.array([100, 1, 2, 50, 30, 6.93, 69.3, 10, 10, 10, 100, 100, 100], dtype=float)
    sigma0 = np.array([(hi - lo) * 0.1 for lo, hi in bounds], dtype=float)

    method_key = method.lower()
    # Backward compatibility for previous API values.
    if method_key in ("differential_evolution", "least_squares"):
        method_key = "cmaes"

    if method_key == "cmaes":
        optimiser_method = pints.CMAES
    elif method_key == "xnes":
        optimiser_method = pints.XNES
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'cmaes' or 'xnes'.")

    print(f"Running optimization using PINTS ({method_key})...")
    controller = pints.OptimisationController(
        cost_function,
        x0,
        sigma0=sigma0,
        boundaries=boundaries,
        method=optimiser_method,
    )
    controller.set_max_iterations(1000)
    controller.set_log_to_screen(True)

    best_params, best_score = controller.run()

    parameter_names = RepressilatorModel._param_names

    print("\nBest-fit parameters:")
    for name, value in zip(parameter_names, best_params):
        print(f"  {name}: {value:.4f}")

    print(f"\nFinal cost (SSR): {best_score:.4e}")

    return best_params


def run_inference_for_cell(
    times: np.ndarray,
    cell_data: Dict[str, List[float]],
    method: str = 'differential_evolution',
) -> Dict[str, any]:
    """
    Run ODE parameter inference for a single cell's time-series data.

    This function prepares observation data from a single cell and runs
    parameter optimization to find the best-fit Repressilator parameters.

    Args:
        times: Array of time points in minutes
        cell_data: Dictionary with keys 'nuclear' and 'cytoplasmic' containing
                   lists/arrays of protein concentration measurements over time
        method: Optimization method ('differential_evolution' or 'least_squares')

    Returns:
        Dictionary with keys:
        - 'times': Input time array
        - 'observations': 2D array of shape (n_times, 2) with nuclear and cytoplasmic data
        - 'best_fit_parameters': Array of 13 fitted parameters
        - 'parameter_names': List of parameter names

    Raises:
        ValueError: If cell_data is missing 'nuclear' or 'cytoplasmic' measurements
    """
    # Prepare observations
    nuclear = np.array(cell_data.get('nuclear', []))
    cytoplasmic = np.array(cell_data.get('cytoplasmic', []))

    if len(nuclear) == 0 or len(cytoplasmic) == 0:
        raise ValueError("Cell data must contain 'nuclear' and 'cytoplasmic' measurements")

    observations = np.column_stack([nuclear, cytoplasmic])

    # Run inference
    best_params = infer_parameters(times, observations, method)

    results = {
        'times': times,
        'observations': observations,
        'best_fit_parameters': best_params,
        'parameter_names': RepressilatorModel._param_names,
    }

    return results
