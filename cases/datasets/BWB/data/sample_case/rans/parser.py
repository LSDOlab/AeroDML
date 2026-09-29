import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from foam_output_parser import parse_cfd_case_directory
parse_cfd_case_directory()