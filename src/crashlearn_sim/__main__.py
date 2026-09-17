"""Launch the competition application with python -m crashlearn_sim."""

from multiprocessing import freeze_support
from crashlearn_sim.app import main

if __name__ == "__main__":
    freeze_support()
    main()
