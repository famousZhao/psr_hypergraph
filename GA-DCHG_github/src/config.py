"""Configuration for the GA-DCHG bearing-fault paper (CWRU only)."""
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
RESULTS = PROJ / "results"
FIGURES = PROJ / "figures"
TABLES = PROJ / "manuscript" / "tables"
CACHE = RESULTS / "cache"
for _p in (RESULTS, FIGURES, TABLES, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- raw data
D_CWRU = Path("D:/数模汇总/CWRU/Data")

# ---------------------------------------------------------------- signal windows
L_WINDOW = 2048        # samples per segment (per training instance)
FS_CWRU = 12000.0

# ---------------------------------------------------------------- phase space
PSR_M = 3              # embedding dimension (3D so curvature is well defined)
PSR_TAU = {"cwru": 4}  # delay by first local minimum of the AMI
N_TRACE = 256          # state points kept per window (uniform decimation)
LAMBDA_T = {"cwru": 16.0}  # time-decay scale in trace-index units

# ---------------------------------------------------------------- hyperedges
K_DYN = 8              # order of the dynamical neighbourhood
KMAX = 12              # cap on hyperedge size (padding length)
N_SEG = 4              # local segments per trace
SEG_LEN = N_TRACE // N_SEG   # 64 state points per local segment
GLOBAL_MAX = 40        # cap on the size of a global hyperedge
GLOBAL_STRIDE = 1      # decimation of the trace used by the global channel
N_EIG = 20             # Laplacian eigenvalues kept by the global channel
N_GLOBAL = 51          # global feature vector length
NODEF_DIM = 3          # speed, dynamical radius, incidence degree

# ---------------------------------------------------------------- model
HIDDEN = 40
LAYERS = 2
DROPOUT = 0.2
EPOCHS = 60
BATCH = 16
LR = 2e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12
PHI_DIM = 3            # curvature, expansion index, filling ratio

# ---------------------------------------------------------------- data splits
TEST_FRAC = 0.20
VAL_FRAC = 0.15
N_PER_CLASS = 100      # windows drawn per class per dataset
SEED = 20261016
