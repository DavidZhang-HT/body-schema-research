"""De-networked dynamic get-up search and static explanations."""
import os

# Set before any NumPy/SciPy import, including the offline explanation entrypoint.
for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
