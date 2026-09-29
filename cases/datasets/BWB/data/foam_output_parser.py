#!/usr/bin/env python3
"""
parse_cfd_logs.py

A small CFD log parser for DAFoam/OpenFOAM case folders.

Expected use:
    Call parse_cfd_case_directory() while your current working directory is
    either the euler/ directory or the rans/ directory.

Expected directory layout:
    euler/ or rans/
        0.npz
        1.npz
        ...
        logs/
            output_logs_0/.../rank.00/stdout
            output_logs_1/.../rank.00/stdout
            ...

Outputs:
    parsed_case_summary.json
    parser_plots/*.png   optional

Design choices:
    - The current working directory is the data directory.
    - No data_dir argument.
    - No CSV output.
    - No parsed fields are written back into .npz files.
    - NaN detection is based only on log lines, not on reading .npz arrays.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple



SUMMARY_JSON_NAME = "parsed_case_summary.json"
PLOTS_DIR_NAME = "parser_plots"
SCHEMA_VERSION = 1

# Residual variables to plot if they appear in the logs.
RESIDUAL_FIELDS_TO_PLOT = ("p", "U0", "U1", "U2", "nuTilda", "he", "T")

FLOAT = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?|nan|inf)"

TIME_RE = re.compile(
    r"^\s*Time\s*=\s*(?P<time>" + FLOAT + r")\s*$",
    re.IGNORECASE,
)

COEFF_RE = re.compile(
    r"^\s*(?P<name>C[dlm])\s*:\s*(?P<total>" + FLOAT + r")\s*"
    r"\(\s*pressure:\s*(?P<pressure>" + FLOAT + r")\s+"
    r"viscous:\s*(?P<viscous>" + FLOAT + r")\s*\)",
    re.IGNORECASE,
)

FORCE_RE = re.compile(
    r"^\s*(?P<name>drag|lift)\s*:\s*(?P<value>" + FLOAT + r")\s+"
    r"final\s*:\s*(?P<final>" + FLOAT + r")",
    re.IGNORECASE,
)

SOLVER_RES_RE = re.compile(
    r"^\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"initRes:\s*(?P<init>" + FLOAT + r")\s+"
    r"finalRes:\s*(?P<final>" + FLOAT + r")\s+"
    r"nIters:\s*(?P<niters>\d+)",
    re.IGNORECASE,
)

FINAL_RES_RE = re.compile(
    r"^\s*(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s+"
    r"Residual\s+(?P<kind>Norm2|Mean|Max):\s*(?P<value>.*)$",
    re.IGNORECASE,
)

SAVED_RE = re.compile(
    r"Saved data to\s+(?P<path>.*?(?P<case>\d+)\.npz),\s*"
    r"lift:\s*\[\s*(?P<lift>" + FLOAT + r")\s*\],\s*"
    r"drag:\s*\[\s*(?P<drag>" + FLOAT + r")\s*\],\s*"
    r"converged:\s*(?P<converged>\d+)",
    re.IGNORECASE,
)

NAN_COUNT_RE = re.compile(
    r"NaN count:\s*(?P<count>\d+)",
    re.IGNORECASE,
)

EVAL_TIME_RE = re.compile(
    r"Simulation\s+(?P<case>\d+)\s+evaluation time:\s+"
    r"(?P<seconds>" + FLOAT + r")",
    re.IGNORECASE,
)


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def to_float(text: str) -> float:
    """Convert a parsed number string into a float."""
    return float(text.strip())


def scalar_or_list(text: str) -> Any:
    """
    Parse text that may contain either one number or several numbers.

    Examples:
        "1.2e-6"  -> 1.2e-6
        "(1 2 3)" -> [1.0, 2.0, 3.0]
    """
    values = [to_float(x) for x in re.findall(FLOAT, text, flags=re.IGNORECASE)]

    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return values


def json_safe(value: Any) -> Any:
    """
    Recursively convert NaN/inf values to None before writing JSON.

    This keeps the JSON easy for other scripts to read.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    return value


def sorted_case_paths(paths: Iterable[Path]) -> List[Path]:
    """
    Sort paths by the last number in the name.

    This makes output_logs_2 come before output_logs_10.
    """
    def key(path: Path) -> Tuple[int, str]:
        numbers = re.findall(r"\d+", path.name)
        number = int(numbers[-1]) if numbers else 10**18
        return number, str(path)

    return sorted(paths, key=key)


def case_id_from_output_dir(path: Path) -> Optional[int]:
    """Extract case ID from a directory named output_logs_123."""
    match = re.search(r"output_logs_(\d+)$", path.name)
    if match is None:
        return None
    return int(match.group(1))


def read_text_file(path: Path) -> str:
    """Read a text file, returning an empty string if something goes wrong."""
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def find_rank0_log_text(output_log_dir: Path) -> str:
    """
    Find and concatenate rank.00 log text for one case.

    We only read rank.00 because that usually contains the case-level values we
    need and avoids duplicating information from other MPI ranks.
    """
    text_parts: List[str] = []

    # Most common structure: output_logs_0/.../rank.00/stdout
    rank_dirs = [p for p in output_log_dir.rglob("rank.00") if p.is_dir()]
    rank_files = [p for p in output_log_dir.rglob("rank.00") if p.is_file()]

    # Sometimes rank.00 itself is a file.
    for rank_file in sorted_case_paths(rank_files):
        text = read_text_file(rank_file)
        if text:
            text_parts.append(text)

    # More commonly, rank.00 is a directory containing stdout.
    for rank_dir in sorted_case_paths(rank_dirs):
        files = [p for p in rank_dir.rglob("*") if p.is_file()]
        files = sorted_case_paths(files)

        # Prefer stdout/log-looking files first.
        files.sort(
            key=lambda p: (
                0 if p.name.lower() in {"stdout", "log", "output.log"} else 1,
                str(p),
            )
        )

        for file_path in files:
            text = read_text_file(file_path)
            if text:
                text_parts.append(text)

    # Fallback: some runs may have stdout files outside rank.00.
    if not text_parts:
        stdout_files = [p for p in output_log_dir.rglob("*stdout*") if p.is_file()]
        for file_path in sorted_case_paths(stdout_files):
            text = read_text_file(file_path)
            if text:
                text_parts.append(text)

    return "\n".join(text_parts)


# -----------------------------------------------------------------------------
# Parsing one case
# -----------------------------------------------------------------------------

def empty_case(case_id: int) -> Dict[str, Any]:
    """Create the full internal dictionary for one case."""
    return {
        "case": case_id,
        "log_found": False,

        # Time histories.
        "times": [],
        "coeff_print_iter": [],

        # Aerodynamic coefficients.
        "Cl": [],
        "Cl_pressure": [],
        "Cl_viscous": [],
        "Cd": [],
        "Cd_pressure": [],
        "Cd_viscous": [],
        "Cm": [],
        "Cm_pressure": [],
        "Cm_viscous": [],

        # Forces.
        "force_time": [],
        "lift": [],
        "drag": [],

        # Solver residual histories:
        # var -> {time, init, final, niters}
        "solver_residuals": {},

        # Final residual stats:
        # var -> {Norm2, Mean, Max}
        "final_residual_stats": {},

        # NaN information from logs only.
        "log_nan_count_total": 0,
        "log_nan_counts": [],

        # Values from the final "Saved data to ..." line.
        "saved_lift": None,
        "saved_drag": None,
        "converged": None,

        # Optional timing information.
        "evaluation_time_seconds": None,
    }


def parse_log_text(text: str, case_id: int) -> Dict[str, Any]:
    """Parse rank.00 text for one case."""
    case = empty_case(case_id)
    case["log_found"] = bool(text.strip())

    current_time: Optional[float] = None
    coeff_print_count = 0

    for line in text.splitlines():
        # Time = ...
        match = TIME_RE.match(line)
        if match:
            current_time = to_float(match.group("time"))
            case["times"].append(current_time)
            continue

        # Example residual line:
        # p initRes: 1e-04 finalRes: 1e-07 nIters: 2
        match = SOLVER_RES_RE.match(line)
        if match:
            var = match.group("var")
            record = case["solver_residuals"].setdefault(
                var,
                {"time": [], "init": [], "final": [], "niters": []},
            )
            record["time"].append(current_time)
            record["init"].append(to_float(match.group("init")))
            record["final"].append(to_float(match.group("final")))
            record["niters"].append(int(match.group("niters")))
            continue

        # Lift/drag line.
        match = FORCE_RE.match(line)
        if match:
            name = match.group("name").lower()
            final_value = to_float(match.group("final"))

            if name == "lift":
                case["force_time"].append(current_time)
                case["lift"].append(final_value)
            elif name == "drag":
                case["drag"].append(final_value)
            continue

        # Coefficient line: Cl, Cd, or Cm.
        match = COEFF_RE.match(line)
        if match:
            name = match.group("name").capitalize()  # Cl, Cd, or Cm
            total = to_float(match.group("total"))
            pressure = to_float(match.group("pressure"))
            viscous = to_float(match.group("viscous"))

            # Usually Cl/Cd/Cm are printed together. Count Cl prints as the
            # coefficient-print iteration index.
            if name == "Cl":
                coeff_print_count += 1
                case["coeff_print_iter"].append(coeff_print_count)

            case[name].append(total)
            case[f"{name}_pressure"].append(pressure)
            case[f"{name}_viscous"].append(viscous)
            continue

        # Final residual stats such as:
        # p Residual Norm2: 1.23e-7
        match = FINAL_RES_RE.match(line)
        if match:
            var = match.group("var")
            kind = match.group("kind")
            value = scalar_or_list(match.group("value"))
            case["final_residual_stats"].setdefault(var, {})[kind] = value
            continue

        # NaN count from logs only.
        match = NAN_COUNT_RE.search(line)
        if match:
            count = int(match.group("count"))
            case["log_nan_counts"].append(count)
            case["log_nan_count_total"] += count
            continue

        # Final saved values.
        match = SAVED_RE.search(line)
        if match:
            case["saved_lift"] = to_float(match.group("lift"))
            case["saved_drag"] = to_float(match.group("drag"))
            case["converged"] = int(match.group("converged"))
            continue

        # Evaluation time.
        match = EVAL_TIME_RE.search(line)
        if match:
            case["evaluation_time_seconds"] = to_float(match.group("seconds"))
            continue

    return case


# -----------------------------------------------------------------------------
# Summaries
# -----------------------------------------------------------------------------

def last_value(case: Dict[str, Any], key: str) -> Optional[float]:
    """Return the last value in a time history, or None if it is empty."""
    values = case.get(key, [])
    return values[-1] if values else None


def last_solver_residual(case: Dict[str, Any], var: str) -> Optional[float]:
    """Return the last finalRes value for a solver variable."""
    record = case.get("solver_residuals", {}).get(var)
    if record and record.get("final"):
        return record["final"][-1]
    return None


def residual_norm2(case: Dict[str, Any], var: str) -> Any:
    """Return final residual Norm2 for a variable, if it was printed."""
    return case.get("final_residual_stats", {}).get(var, {}).get("Norm2")


def final_solver_residuals(case: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Return the last finalRes value for every solver variable found."""
    output: Dict[str, Optional[float]] = {}
    for var in case.get("solver_residuals", {}):
        output[var] = last_solver_residual(case, var)
    return output


def add_final_force_points(case: Dict[str, Any]) -> None:
    """
    Add saved final lift/drag to the internal force histories for plotting.

    This does not change the JSON logic. It only makes the force plots include
    the final saved point when available.
    """
    saved_lift = case.get("saved_lift")
    saved_drag = case.get("saved_drag")

    if saved_lift is None and saved_drag is None:
        return

    if case.get("force_time"):
        final_x = case["force_time"][-1]
    elif case.get("times"):
        final_x = case["times"][-1]
    else:
        final_x = len(case.get("Cl", []))

    if saved_lift is not None:
        if not case.get("lift") or case["lift"][-1] != saved_lift:
            case["force_time"].append(final_x)
            case["lift"].append(saved_lift)

            if saved_drag is not None:
                case["drag"].append(saved_drag)

    elif saved_drag is not None:
        if not case.get("drag") or case["drag"][-1] != saved_drag:
            case["force_time"].append(final_x)
            case["drag"].append(saved_drag)


def compact_case_summary(case: Dict[str, Any], data_dir: Path) -> Dict[str, Any]:
    """Build the compact per-case dictionary that gets written to JSON."""
    case_id = case["case"]
    npz_path = data_dir / f"{case_id}.npz"

    final_lift = (
        case.get("saved_lift")
        if case.get("saved_lift") is not None
        else last_value(case, "lift")
    )
    final_drag = (
        case.get("saved_drag")
        if case.get("saved_drag") is not None
        else last_value(case, "drag")
    )

    summary = {
        "schema_version": SCHEMA_VERSION,

        # Solver is inferred from the current directory name, e.g. euler/rans.
        "solver": data_dir.name,

        # Case identity.
        "case": case_id,
        "npz_name": f"{case_id}.npz",

        # Availability checks.
        "log_found": case.get("log_found", False),
        "npz_exists": npz_path.exists(),

        # NaN status from logs only.
        "nans_exist": case.get("log_nan_count_total", 0) > 0,
        "log_nan_count_total": case.get("log_nan_count_total", 0),
        "log_nan_counts": case.get("log_nan_counts", []),

        # Iteration and time information.
        "n_iterations": len(case.get("Cl", [])),
        "n_coeff_prints": len(case.get("Cl", [])),
        "last_openfoam_time": last_value(case, "times"),
        "n_force_samples": len(case.get("lift", [])),

        # Residuals.
        "p_final_residual": last_solver_residual(case, "p"),
        "p_residual_norm2": residual_norm2(case, "p"),
        "U_residual_norm2": residual_norm2(case, "U"),
        "final_solver_residuals": final_solver_residuals(case),

        # Coefficients.
        "final_Cl": last_value(case, "Cl"),
        "final_Cd": last_value(case, "Cd"),
        "final_Cm": last_value(case, "Cm"),

        # Forces.
        "final_L": final_lift,
        "final_D": final_drag,

        # Coefficient components.
        "pressure_lift_coeff": last_value(case, "Cl_pressure"),
        "viscous_lift_coeff": last_value(case, "Cl_viscous"),
        "pressure_drag_coeff": last_value(case, "Cd_pressure"),
        "viscous_drag_coeff": last_value(case, "Cd_viscous"),

        # Solver-reported status.
        "converged": case.get("converged"),
        "evaluation_time_seconds": case.get("evaluation_time_seconds"),
    }

    return json_safe(summary)


# -----------------------------------------------------------------------------
# Terminal table output
# -----------------------------------------------------------------------------

def format_table_value(value: Any, width: int = 12, precision: int = 5) -> str:
    """Format one value for the terminal summary table."""
    if value is None:
        return "None".rjust(width)
    if isinstance(value, float):
        return f"{value:{width}.{precision}g}"
    return str(value).rjust(width)


def print_case_table(summaries: List[Dict[str, Any]], title: str = "Parsed case summary") -> None:
    """Print a compact terminal table like the original script did."""
    if not summaries:
        print(f"\n{title}: no cases to show")
        return

    print(f"\n{title}")
    print("-" * 132)
    print(
        f"{'case':>5}  {'iters':>6}  {'lastTime':>8}  {'nans':>5}  {'conv':>4}  "
        f"{'final Cl':>12}  {'final Cd':>12}  {'final L':>14}  {'final D':>14}  {'p finalRes':>12}"
    )
    print("-" * 132)

    for summary in summaries:
        print(
            f"{summary['case']:5d}  "
            f"{summary['n_iterations']:6d}  "
            f"{format_table_value(summary.get('last_openfoam_time'), 8, 5)}  "
            f"{str(summary.get('nans_exist')):>5}  "
            f"{format_table_value(summary.get('converged'), 4)}  "
            f"{format_table_value(summary.get('final_Cl'))}  "
            f"{format_table_value(summary.get('final_Cd'))}  "
            f"{format_table_value(summary.get('final_L'), 14)}  "
            f"{format_table_value(summary.get('final_D'), 14)}  "
            f"{format_table_value(summary.get('p_final_residual'))}"
        )

    print("-" * 132)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def case_is_safe_for_plots(
    case: Dict[str, Any],
    residual_plot_max: Optional[float],
) -> bool:
    """
    Decide whether a case should be included in plots.

    If residual_plot_max is None or math.inf, no residual cutoff is applied.
    Otherwise, a case is excluded if any final solver residual is non-finite or
    larger than residual_plot_max.
    """
    if residual_plot_max is None or math.isinf(residual_plot_max):
        return True

    for record in case.get("solver_residuals", {}).values():
        final_values = record.get("final", [])
        if not final_values:
            continue

        last_final = final_values[-1]

        if not math.isfinite(last_final):
            return False
        if last_final > residual_plot_max:
            return False

    return True


def plot_case_history(
    cases: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    logy: bool = False,
) -> None:
    """Plot one time-history quantity for all accepted cases."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_lines = False
    show_legend = len(cases) <= 20

    for case in cases:
        x = case.get(x_key, [])
        y = case.get(y_key, [])

        if not x or not y:
            continue

        n = min(len(x), len(y))
        if n == 0:
            continue

        ax.plot(
            x[:n],
            y[:n],
            linewidth=1.0,
            alpha=0.75,
            label=f"case {case['case']}" if show_legend else None,
        )
        any_lines = True

    if not any_lines:
        plt.close(fig)
        return

    ax.set_title(title)
    ax.set_xlabel("coefficient print iteration" if x_key == "coeff_print_iter" else "OpenFOAM time")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    if logy:
        ax.set_yscale("log")

    if show_legend:
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_residual_history(
    cases: List[Dict[str, Any]],
    var: str,
    output_path: Path,
) -> None:
    """Plot final residual history for one residual variable."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5.5))
    any_lines = False
    show_legend = len(cases) <= 20

    for case in cases:
        record = case.get("solver_residuals", {}).get(var)
        if not record:
            continue

        x = record.get("time", [])
        y = record.get("final", [])

        if not x or not y:
            continue

        n = min(len(x), len(y))
        if n == 0:
            continue

        ax.plot(
            x[:n],
            y[:n],
            linewidth=1.0,
            alpha=0.75,
            label=f"case {case['case']}" if show_legend else None,
        )
        any_lines = True

    if not any_lines:
        plt.close(fig)
        return

    ax.set_title(f"{var} final residual vs OpenFOAM time")
    ax.set_xlabel("OpenFOAM time")
    ax.set_ylabel(f"{var} finalRes")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)

    if show_legend:
        ax.legend(fontsize=8)

    # ax.set_xlim(-30, 1030)
    # ax.set_ylim(4e-9, 0.17)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_final_residuals_by_case(
    cases: List[Dict[str, Any]],
    residual_plot_max: Optional[float],
    output_path: Path,
) -> None:
    """Plot each case's largest final solver residual, colored by plot inclusion."""
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    case_ids: List[int] = []
    residuals: List[float] = []
    colors: List[str] = []

    for case in cases:
        final_values = []
        for record in case.get("solver_residuals", {}).values():
            values = record.get("final", [])
            if values:
                final_values.append(values[-1])

        finite_values = [value for value in final_values if math.isfinite(value)]
        if not finite_values:
            continue

        case_ids.append(case["case"])
        residuals.append(max(finite_values))
        colors.append(
            "tab:blue"
            if case_is_safe_for_plots(case, residual_plot_max)
            else "tab:red"
        )

    if not case_ids:
        return

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar([str(case_id) for case_id in case_ids], residuals, color=colors, alpha=0.85)

    legend_handles = [
        Patch(facecolor="tab:blue", label="included"),
        Patch(facecolor="tab:red", label="excluded"),
    ]

    if residual_plot_max is not None and math.isfinite(residual_plot_max):
        ax.axhline(
            residual_plot_max,
            color="black",
            linestyle="--",
            linewidth=1.0,
            label=f"residual cutoff= {residual_plot_max:g}",
        )
        legend_handles.append(ax.lines[-1])

    ax.legend(handles=legend_handles, fontsize=8)

    ax.set_title("Largest final residual")
    ax.set_xlabel("case")
    ax.set_ylabel("largest finalRes")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_parser_plots(
    cases: List[Dict[str, Any]],
    data_dir: Path,
    residual_plot_max: Optional[float],
    plots_dir_name: str,
) -> List[int]:
    """
    Write parser plots and return the case IDs excluded from plotting.

    The JSON always includes all cases. The cutoff only affects plots.
    """
    plots_dir = data_dir / plots_dir_name
    plots_dir.mkdir(exist_ok=True)

    plot_cases = [case for case in cases if case_is_safe_for_plots(case, residual_plot_max)]
    excluded_case_ids = [case["case"] for case in cases if not case_is_safe_for_plots(case, residual_plot_max)]

    plot_final_residuals_by_case(
        cases,
        residual_plot_max,
        plots_dir / "final_residuals_by_case.png",
    )

    plot_case_history(
        plot_cases,
        "coeff_print_iter",
        "Cl",
        "Cl vs coefficient print iteration",
        "Cl",
        plots_dir / "Cl_vs_print_iteration.png",
    )

    plot_case_history(
        plot_cases,
        "coeff_print_iter",
        "Cd",
        "Cd vs coefficient print iteration",
        "Cd",
        plots_dir / "Cd_vs_print_iteration.png",
    )

    plot_case_history(
        plot_cases,
        "force_time",
        "lift",
        "Lift vs OpenFOAM time",
        "Lift",
        plots_dir / "L_vs_openfoam_time.png",
    )

    plot_case_history(
        plot_cases,
        "force_time",
        "drag",
        "Drag vs OpenFOAM time",
        "Drag",
        plots_dir / "D_vs_openfoam_time.png",
    )

    plot_case_history(
        plot_cases,
        "coeff_print_iter",
        "Cl_pressure",
        "Cl pressure component vs print iteration",
        "Cl pressure",
        plots_dir / "Cl_pressure_vs_print_iteration.png",
    )

    plot_case_history(
        plot_cases,
        "coeff_print_iter",
        "Cl_viscous",
        "Cl viscous component vs print iteration",
        "Cl viscous",
        plots_dir / "Cl_viscous_vs_print_iteration.png",
    )

    plot_case_history(
        plot_cases,
        "coeff_print_iter",
        "Cd_pressure",
        "Cd pressure component vs print iteration",
        "Cd pressure",
        plots_dir / "Cd_pressure_vs_print_iteration.png",
    )

    plot_case_history(
        plot_cases,
        "coeff_print_iter",
        "Cd_viscous",
        "Cd viscous component vs print iteration",
        "Cd viscous",
        plots_dir / "Cd_viscous_vs_print_iteration.png",
    )

    for var in RESIDUAL_FIELDS_TO_PLOT:
        plot_residual_history(
            plot_cases,
            var,
            plots_dir / f"residual_{var}_finalRes_vs_openfoam_time.png",
        )

    return excluded_case_ids


# -----------------------------------------------------------------------------
# Main function to call from euler/ or rans/
# -----------------------------------------------------------------------------

def parse_cfd_case_directory(
    *,
    residual_plot_max: Optional[float] = 3e-6,
    make_plots: bool = True,
    print_summary: bool = True,
    summary_json_name: str = SUMMARY_JSON_NAME,
    plots_dir_name: str = PLOTS_DIR_NAME,
) -> List[Dict[str, Any]]:
    """
    Parse CFD logs in the current working directory.

    Args:
        residual_plot_max:
            Maximum allowed final solver residual for including a case in plots.
            This only affects plots. All cases are still saved in the JSON.
            Use None or math.inf to disable the plot cutoff.

        make_plots:
            If True, write plots into parser_plots/.

        print_summary:
            If True, print terminal tables after parsing.

        summary_json_name:
            Name of the JSON summary file written inside the current directory.

        plots_dir_name:
            Name of the plot output directory inside the current directory.

    Returns:
        The compact summaries that were also written to JSON.
    """
    data_dir = Path.cwd()
    logs_dir = data_dir / "logs"

    if not logs_dir.exists():
        raise FileNotFoundError(
            f"Could not find logs directory: {logs_dir}\n"
            "Call parse_cfd_case_directory() from inside an euler/ or rans/ directory."
        )

    output_dirs = sorted_case_paths(
        p for p in logs_dir.glob("output_logs_*") if p.is_dir()
    )

    if not output_dirs:
        raise FileNotFoundError(
            f"No output_logs_* directories found under: {logs_dir}"
        )

    cases: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []

    for index, output_dir in enumerate(output_dirs, start=1):
        case_id = case_id_from_output_dir(output_dir)
        if case_id is None:
            continue

        print(f"Parsing case {case_id} ({index}/{len(output_dirs)})")

        log_text = find_rank0_log_text(output_dir)
        case = parse_log_text(log_text, case_id)

        if not case["log_found"]:
            print(f"  Warning: no rank.00 log text found for case {case_id}")

        add_final_force_points(case)

        cases.append(case)
        summaries.append(compact_case_summary(case, data_dir))

    cases.sort(key=lambda case: case["case"])
    summaries.sort(key=lambda summary: summary["case"])

    output_json_path = data_dir / summary_json_name
    with output_json_path.open("w") as file:
        json.dump(summaries, file, indent=2)

    excluded_plot_case_ids: List[int] = []
    if make_plots:
        excluded_plot_case_ids = make_parser_plots(
            cases=cases,
            data_dir=data_dir,
            residual_plot_max=residual_plot_max,
            plots_dir_name=plots_dir_name,
        )

    if print_summary:
        print_case_table(summaries, title="Parsed case summary")

        if make_plots:
            if excluded_plot_case_ids:
                excluded_summaries = [
                    summary for summary in summaries
                    if summary["case"] in set(excluded_plot_case_ids)
                ]
                print(
                    f"\nExcluded {len(excluded_plot_case_ids)} case(s) from plots "
                    f"because a final solver residual was above {residual_plot_max}: "
                    f"{excluded_plot_case_ids}"
                )
                print_case_table(excluded_summaries, title="Cases excluded from plots")
            else:
                print(f"\nNo cases were excluded from plots by residual_plot_max={residual_plot_max}.")

    print(f"\nParsed {len(summaries)} cases.")
    print(f"Wrote JSON summary to: {output_json_path}")

    if make_plots:
        print(f"Wrote plots to: {data_dir / plots_dir_name}")

    return summaries
