import multiprocessing
from library.launcher import run_worker


def main():
    # Cite debug_lesson_8: Isolate Heavy Workloads in Subprocesses for Reliable Cleanup
    # Use multiprocessing with 'spawn' context to ensure a fresh CUDA state,
    # preventing OOM errors caused by zombie references in the persistent environment.
    # Cite debug_lesson_9: Externalize Multiprocessing Targets When Using `spawn` in Interactive Environments
    ctx = multiprocessing.get_context("spawn")
    p = ctx.Process(target=run_worker)
    p.start()
    p.join()

    if p.exitcode != 0:
        raise RuntimeError(f"Training process failed with exit code {p.exitcode}")


if __name__ == "__main__":
    main()
