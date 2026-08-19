"""
Registration job specs and execution backends for SMINT.

Registration runs in a **separate process** from any GUI that launches it. That
is not incidental: napari and ``napari_spatialdata`` live in an environment with
numpy>=2, where STalign cannot be imported at all, while STalign needs numpy<2.
Passing a job spec on disk to a worker interpreter keeps the two apart, lets a
failed registration fail without taking the GUI down, and makes the batch and
local paths differ only in how the worker is launched.

Flow::

    JobSpec  --write_spec-->  spec.json
                                  |
                    sbatch / subprocess launches
                                  |
                  <worker python> -m smint.cli.register spec.json
                                  |
                          status.json + outputs

The worker interpreter must be one where STalign imports (numpy<2) for the
``st_sm`` round; the ``centroid`` round has no STalign dependency and runs
anywhere.
"""

import json
import logging
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

ROUND_ST_SM = "st_sm"
ROUND_CENTROID = "centroid"
VALID_ROUNDS = (ROUND_ST_SM, ROUND_CENTROID)

VALID_BACKENDS = ("sbatch", "local")

#: Terminal states; anything else means the job is still in flight.
TERMINAL_STATES = ("completed", "failed", "cancelled")

#: Environment variables that must not leak from the caller into the worker.
#:
#: The whole point of running the worker out-of-process is that the GUI lives
#: in a numpy>=2 environment while STalign needs numpy<2. Inheriting the
#: caller's ``PYTHONPATH``/``PYTHONHOME`` would put the GUI's site-packages
#: back on the worker's import path, and an inherited loader path can change
#: which shared objects torch resolves at runtime -- both of which turn the
#: worker's behaviour into a function of how the GUI happened to be launched.
LEAKY_ENV_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "LD_PRELOAD")


def worker_env(base: Optional[dict] = None) -> dict:
    """Return a copy of ``base`` (default ``os.environ``) with :data:`LEAKY_ENV_VARS` dropped."""
    env = dict(os.environ if base is None else base)
    for name in LEAKY_ENV_VARS:
        env.pop(name, None)
    return env


def available_partitions() -> list:
    """
    Partition names this cluster accepts, via ``sinfo``.

    Returns an empty list when SLURM is unavailable, so callers can fall back
    rather than blocking on a check they cannot perform.
    """
    if not shutil.which("sinfo"):
        return []
    try:
        result = subprocess.run(
            ["sinfo", "-h", "-o", "%P"], capture_output=True, text=True, timeout=30
        )
    except (subprocess.SubprocessError, OSError):
        return []
    if result.returncode != 0:
        return []
    # sinfo marks the default partition with a trailing '*'
    return sorted({line.strip().rstrip("*") for line in result.stdout.split() if line.strip()})


def default_worker_python() -> str:
    """
    Best guess at an interpreter that can run registration.

    Order: ``SMINT_WORKER_PYTHON``, then ``STalign_env/bin/python`` beside the
    installed package, then the current interpreter.
    """
    env = os.environ.get("SMINT_WORKER_PYTHON")
    if env:
        return env

    # smint/alignment/jobs.py -> smint/alignment -> smint -> SMINT -> project
    project_root = Path(__file__).resolve().parents[3]
    candidate = project_root / "STalign_env" / "bin" / "python"
    if candidate.exists():
        return str(candidate)

    import sys
    return sys.executable


@dataclass
class SlurmResources:
    """
    SLURM resource request for a registration job.

    Set ``gpus`` to request GPUs, and remember to point ``partition`` at one
    that actually has them -- asking for a GPU on a CPU partition leaves the
    job pending indefinitely rather than failing, which is easy to mistake for
    a slow queue. :meth:`check` warns about that combination.
    """

    partition: str = "regular"
    cpus_per_task: int = 8
    memory: str = "64G"
    time_limit: str = "04:00:00"
    job_name: str = "smint_register"
    gpus: int = 0

    #: Substrings that identify a GPU partition on this cluster.
    GPU_PARTITION_HINTS = ("gpu", "a100", "a30", "p100", "a10")

    def check(self) -> None:
        """
        Validate the resource request before anything reaches ``sbatch``.

        Raises
        ------
        ValueError
            If the partition is not one this cluster offers. Catching it here
            turns SLURM's terse "Invalid partition name specified" into a
            message naming the valid options.
        """
        partitions = available_partitions()
        if partitions and self.partition not in partitions:
            raise ValueError(
                f"Partition {self.partition!r} does not exist on this cluster. "
                f"Available: {', '.join(partitions)}"
            )

        if self.gpus and not any(
            hint in self.partition.lower() for hint in self.GPU_PARTITION_HINTS
        ):
            logger.warning(
                "Requested %d GPU(s) on partition %r, which does not look like "
                "a GPU partition. The job will sit pending rather than fail. "
                "Use a GPU partition (e.g. 'gpuq') or set gpus=0.",
                self.gpus, self.partition,
            )
        if not self.gpus and any(
            hint in self.partition.lower() for hint in self.GPU_PARTITION_HINTS
        ):
            logger.warning(
                "Partition %r looks like a GPU partition but gpus=0, so no GPU "
                "will be allocated and registration will run on CPU.",
                self.partition,
            )

    def sbatch_directives(self, log_dir: Path) -> str:
        lines = [
            f"#SBATCH --job-name={self.job_name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --cpus-per-task={self.cpus_per_task}",
            f"#SBATCH --mem={self.memory}",
            f"#SBATCH --time={self.time_limit}",
            f"#SBATCH --output={log_dir / (self.job_name + '_%j.out')}",
            f"#SBATCH --error={log_dir / (self.job_name + '_%j.err')}",
        ]
        if self.gpus:
            lines.append(f"#SBATCH --gres=gpu:{self.gpus}")
        return "\n".join(lines)


@dataclass
class JobSpec:
    """
    A registration job, serialisable to JSON.

    Parameters
    ----------
    round : {'st_sm', 'centroid'}
        Which registration to run. ``st_sm`` is the STalign LDDMM path for
        sequential sections; ``centroid`` is the correspondence-based path for
        post-staining on the same section.
    inputs : dict
        Input paths. For ``st_sm``: ``st_file``, ``sm_file``, ``st_points``,
        ``sm_points``. For ``centroid``: ``source_file``, ``target_file``.
    params : dict
        Keyword arguments forwarded to the registration function.
    output_dir : str
        Directory for outputs, ``status.json`` and logs.
    backend : {'sbatch', 'local'}
        How to launch the worker.
    resources : SlurmResources
        Ignored when ``backend='local'``.
    worker_python : str, optional
        Interpreter for the worker; defaults to :func:`default_worker_python`.
    """

    round: str
    inputs: dict
    output_dir: str
    params: dict = field(default_factory=dict)
    backend: str = "sbatch"
    resources: SlurmResources = field(default_factory=SlurmResources)
    worker_python: Optional[str] = None
    created: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    def __post_init__(self):
        if self.round not in VALID_ROUNDS:
            raise ValueError(f"round must be one of {VALID_ROUNDS}, got {self.round!r}")
        if self.backend not in VALID_BACKENDS:
            raise ValueError(
                f"backend must be one of {VALID_BACKENDS}, got {self.backend!r}"
            )
        if isinstance(self.resources, dict):
            self.resources = SlurmResources(**self.resources)
        if self.worker_python is None:
            self.worker_python = default_worker_python()

    # -- required inputs per round, checked before we bother launching --
    REQUIRED_INPUTS = {
        ROUND_ST_SM: ("st_file", "sm_file", "st_points", "sm_points"),
        ROUND_CENTROID: ("source_file", "target_file"),
    }

    #: Path prefixes that are node-local on a typical HPC node. A batch job
    #: lands on a different machine, so anything here is invisible to it.
    NODE_LOCAL_PREFIXES = ("/tmp", "/var/tmp", "/dev/shm", "/run")

    def validate(self) -> None:
        """
        Check required inputs are present, exist, and are reachable by the worker.

        Raises
        ------
        ValueError
            If a required key is missing, or an sbatch job references
            node-local paths the compute node cannot see.
        FileNotFoundError
            If a named input file does not exist. Catching this here means a
            typo surfaces immediately instead of after a queue wait.
        """
        required = self.REQUIRED_INPUTS[self.round]
        missing = [k for k in required if not self.inputs.get(k)]
        if missing:
            raise ValueError(
                f"round {self.round!r} requires inputs {missing} which are missing"
            )
        for key in required:
            path = Path(self.inputs[key])
            if not path.exists():
                raise FileNotFoundError(f"{key} does not exist: {path}")

        if self.backend == "sbatch":
            self.resources.check()

            # A batch job runs on another machine: node-local paths silently
            # vanish, and the failure mode is opaque (no logs are written
            # because the log directory itself does not exist there).
            offending = {
                key: str(value)
                for key, value in [("output_dir", self.output_dir), *self.inputs.items()]
                if str(value).startswith(self.NODE_LOCAL_PREFIXES)
            }
            if offending:
                raise ValueError(
                    "sbatch jobs run on a different machine, so node-local "
                    f"paths are unreachable: {offending}. Use a shared "
                    "filesystem (e.g. /vast/scratch or /vast/projects), or "
                    "backend='local' to run here."
                )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["resources"] = asdict(self.resources) if not isinstance(
            self.resources, dict
        ) else self.resources
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "JobSpec":
        data = dict(data)
        data.pop("created", None)
        return cls(**data)


def write_spec(spec: JobSpec, path: Optional[str] = None) -> Path:
    """Serialise a job spec to JSON; defaults to ``<output_dir>/job_spec.json``."""
    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = Path(path) if path else out_dir / "job_spec.json"
    target.write_text(json.dumps(spec.to_dict(), indent=2))
    logger.info("Wrote job spec to %s", target)
    return target


def read_spec(path: str) -> JobSpec:
    """Load a job spec from JSON."""
    return JobSpec.from_dict(json.loads(Path(path).read_text()))


# --------------------------------------------------------------------------
# Status
# --------------------------------------------------------------------------

def status_path(output_dir: str) -> Path:
    return Path(output_dir) / "status.json"


def write_status(output_dir: str, state: str, message: str = "", **extra) -> Path:
    """Write the worker's status file. Called by the worker, read by the GUI."""
    path = status_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state": state,
        "message": message,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    payload.update(extra)
    path.write_text(json.dumps(payload, indent=2))
    return path


def read_status(output_dir: str) -> dict:
    """
    Read a job's status.

    Returns a dict with at least ``state``. ``unknown`` means no status file
    has appeared yet -- normal while a job is queued.
    """
    path = status_path(output_dir)
    if not path.exists():
        return {"state": "unknown", "message": "no status file yet"}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        # The worker may be mid-write; that is transient, not an error.
        return {"state": "unknown", "message": "status file unreadable (mid-write?)"}


def slurm_state(job_id: str) -> Optional[str]:
    """
    Query SLURM for a job's state, or None if SLURM is unavailable.

    Tries ``squeue`` first (job still in the queue), then ``sacct`` (finished).
    """
    if shutil.which("squeue"):
        try:
            out = subprocess.run(
                ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().split()[0]
        except (subprocess.SubprocessError, OSError):
            pass

    if shutil.which("sacct"):
        try:
            out = subprocess.run(
                ["sacct", "-j", str(job_id), "-n", "-X", "-o", "State"],
                capture_output=True, text=True, timeout=30,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        except (subprocess.SubprocessError, OSError):
            pass
    return None


# --------------------------------------------------------------------------
# Launchers
# --------------------------------------------------------------------------

def build_sbatch_script(spec: JobSpec, spec_path: Path) -> str:
    """Render the sbatch script that runs the worker for ``spec``."""
    log_dir = Path(spec.output_dir)
    return f"""#!/bin/bash
{spec.resources.sbatch_directives(log_dir)}

# Generated by smint.alignment.jobs -- do not edit; regenerate from the JobSpec.
set -euo pipefail

# sbatch exports the submitting process's environment, which for a GUI-driven
# submission is napari's. Drop what would otherwise reach into the worker's
# interpreter or its dynamic loader.
unset {' '.join(LEAKY_ENV_VARS)}

export OMP_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{spec.resources.cpus_per_task}}}"
export MKL_NUM_THREADS="${{SLURM_CPUS_PER_TASK:-{spec.resources.cpus_per_task}}}"

echo "host: $(hostname)"
echo "started: $(date)"

{spec.worker_python} -m smint.cli.register {spec_path}

echo "finished: $(date)"
"""


def submit(spec: JobSpec) -> dict:
    """
    Launch a registration job.

    Validates the spec, writes it to disk, then launches the worker via the
    requested backend.

    Parameters
    ----------
    spec : JobSpec

    Returns
    -------
    dict
        ``backend``, ``output_dir``, ``spec_path``, and either ``job_id``
        (sbatch) or ``pid`` (local).

    Raises
    ------
    RuntimeError
        If ``sbatch`` is requested but unavailable, or submission fails.
    """
    spec.validate()
    out_dir = Path(spec.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    spec_path = write_spec(spec)

    write_status(spec.output_dir, "submitted", f"via {spec.backend}")

    if spec.backend == "sbatch":
        if not shutil.which("sbatch"):
            raise RuntimeError(
                "sbatch not found on PATH. Use backend='local' to run in a "
                "subprocess on this machine instead."
            )
        script_path = out_dir / "submit.sh"
        script_path.write_text(build_sbatch_script(spec, spec_path))
        script_path.chmod(0o755)

        result = subprocess.run(
            ["sbatch", str(script_path)], capture_output=True, text=True
        )
        if result.returncode != 0:
            write_status(spec.output_dir, "failed", f"sbatch failed: {result.stderr}")
            raise RuntimeError(f"sbatch submission failed: {result.stderr.strip()}")

        # "Submitted batch job 12345"
        job_id = result.stdout.strip().split()[-1]
        write_status(spec.output_dir, "queued", f"SLURM job {job_id}", job_id=job_id)
        logger.info("Submitted SLURM job %s for %s round", job_id, spec.round)
        return {
            "backend": "sbatch",
            "job_id": job_id,
            "output_dir": str(out_dir),
            "spec_path": str(spec_path),
        }

    # local: detached subprocess so a GUI caller is never blocked
    log_file = out_dir / "worker.log"
    handle = open(log_file, "w")
    process = subprocess.Popen(
        [spec.worker_python, "-m", "smint.cli.register", str(spec_path)],
        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
        env=worker_env(),
    )
    write_status(spec.output_dir, "running", f"local pid {process.pid}", pid=process.pid)
    logger.info("Started local worker pid %s for %s round", process.pid, spec.round)
    return {
        "backend": "local",
        "pid": process.pid,
        "output_dir": str(out_dir),
        "spec_path": str(spec_path),
    }


def poll(output_dir: str) -> dict:
    """
    Current job status, reconciling the status file with SLURM.

    The worker's own status file is authoritative once it exists. SLURM is
    consulted to catch jobs killed before the worker could report -- an OOM or
    timeout leaves the status file saying "running" forever otherwise.

    Returns
    -------
    dict
        ``state`` plus whatever the worker recorded; ``slurm_state`` when known.
    """
    status = read_status(output_dir)
    job_id = status.get("job_id")

    if job_id and status.get("state") not in TERMINAL_STATES:
        sstate = slurm_state(job_id)
        if sstate:
            status["slurm_state"] = sstate
            if sstate.startswith(("FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL")):
                status["state"] = "failed"
                status["message"] = f"SLURM reported {sstate}"
            elif sstate.startswith("COMPLETED") and status.get("state") != "completed":
                # SLURM finished but the worker never wrote success.
                status["state"] = "failed"
                status["message"] = (
                    "SLURM completed but the worker did not report success; "
                    "check the .err log"
                )
    return status
