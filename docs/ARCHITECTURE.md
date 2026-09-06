# Architecture

The package uses a `src/` layout. `hardware` discovers capabilities; `data` creates immutable data
identity; `models` supplies dense and sparse workloads; `parallel` applies TP before bottom-up
FSDP2 and creates optimizers afterward; `profiling` and `metrics` derive rank-local evidence;
`diagnosis` labels bottlenecks; `tuning` exposes SEARCH-only capability views; `recovery` records
distributed and application state; `experiments` produces immutable run directories.

The boundary is deliberate: planners cannot access EVALUATION rows, TorchTitan is isolated under
`external/`, and Windows capability gates cannot be overridden by a Gloo run.
