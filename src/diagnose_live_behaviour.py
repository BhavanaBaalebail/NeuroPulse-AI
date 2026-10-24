"""Write data/live_behaviour_diagnosis.json. Does not retrain or alter live features."""

from behaviour_diagnostics import write_report


if __name__ == "__main__":
    path = write_report()
    print("Wrote", path)
