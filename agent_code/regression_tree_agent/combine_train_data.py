from agent_code.ij_1.data_collector import DataCollector
from pathlib import Path

if __name__ == '__main__':
    DataCollector.combine([str(f) for f in Path("training_data").glob("*.json")], "combined_data.json")