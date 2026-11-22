"""
Environment Verification and System Diagnostic Script
Authored by: Anhum (15% Component)
Checks Python environment, essential libraries, directory permissions, and configuration.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REQUIRED_PACKAGES = [
    ("numpy", "Core numerical computing"),
    ("pandas", "Time-series and tabular manipulation"),
    ("scipy", "Statistical computations and distributions"),
    ("sklearn", "Machine learning algorithms and metrics"),
    ("yaml", "Configuration parsing (pyyaml)"),
    ("matplotlib", "Core plotting and figure generation"),
    ("pytest", "Automated unit testing"),
]

OPTIONAL_PACKAGES = [
    ("seaborn", "Statistical data visualization"),
    ("yfinance", "Historical market data extraction"),
    ("xgboost", "Gradient boosting algorithms"),
    ("hmmlearn", "Hidden Markov Model regime detection"),
    ("shap", "Model interpretability and SHAP values"),
]

REQUIRED_DIRECTORIES = [
    "config",
    "data/raw",
    "data/processed",
    "src/data",
    "src/features",
    "src/models",
    "src/strategy",
    "src/backtest",
    "src/utils",
    "notebooks",
    "reports/figures",
    "tests",
]


def check_python_version() -> bool:
    print("=" * 60)
    print("1. PYTHON RUNTIME CHECK")
    print("=" * 60)
    v = sys.version_info
    print(f"Detected Python Version: {v.major}.{v.minor}.{v.micro}")
    if v.major >= 3 and v.minor >= 10:
        print("[PASS] Python version >= 3.10 is satisfied.\n")
        return True
    else:
        print("[FAIL] Python 3.10+ is recommended.\n")
        return False


def check_packages(pkg_list, label: str) -> bool:
    print("=" * 60)
    print(f"2. {label.upper()} PACKAGES")
    print("=" * 60)
    all_ok = True
    for mod_name, purpose in pkg_list:
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK]   {mod_name:<12} (v{ver:<10}) - {purpose}")
        except ImportError:
            print(f"  [FAIL] {mod_name:<12} (NOT FOUND  ) - {purpose}")
            all_ok = False
    print()
    return all_ok


def check_directories() -> bool:
    print("=" * 60)
    print("3. DIRECTORY STRUCTURE & WRITE PERMISSIONS")
    print("=" * 60)
    root = Path(".")
    all_ok = True
    for d in REQUIRED_DIRECTORIES:
        p = root / d
        p.mkdir(parents=True, exist_ok=True)
        # Test write permission
        test_file = p / ".perm_check"
        try:
            test_file.touch()
            test_file.unlink()
            print(f"  [OK] {d:<25} - exists and writeable")
        except Exception as e:
            print(f"  [FAIL] {d:<25} - permission error: {e}")
            all_ok = False
    print()
    return all_ok


def check_config_file() -> bool:
    print("=" * 60)
    print("4. CONFIGURATION FILE INTEGRITY")
    print("=" * 60)
    from src.utils.config_parser import ConfigParser

    try:
        cfg = ConfigParser("config/config.yaml")
        name = cfg.get("project", "name")
        assets = cfg.get("assets", "equities")
        capital = cfg.get("backtest", "initial_capital")
        print(f"  Project Name:     {name}")
        print(f"  Tracked Equities: {assets}")
        print(f"  Initial Capital:  ${capital:,.2f}")
        print("[PASS] Configuration loaded and parsed successfully.\n")
        return True
    except Exception as e:
        print(f"[FAIL] Configuration failed to load: {e}\n")
        return False


def main() -> None:
    print("\n" + "#" * 60)
    print(" QUANTITATIVE TRADING SYSTEM - DAY 1 DIAGNOSTIC HARNESS")
    print("#" * 60 + "\n")

    py_ok = check_python_version()
    req_ok = check_packages(REQUIRED_PACKAGES, "Required Core")
    check_packages(OPTIONAL_PACKAGES, "Downstream / Optional (Phases 2-4)")
    dir_ok = check_directories()
    cfg_ok = check_config_file()

    print("=" * 60)
    print("FINAL SYSTEM STATUS:")
    if py_ok and req_ok and dir_ok and cfg_ok:
        print("  >>> ALL DAY 1 ENVIRONMENT & INTEGRATION CHECKS PASSED! <<<")
    else:
        print("  >>> SYSTEM CHECKS COMPLETED WITH WARNINGS/MISSING ITEMS <<<")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
