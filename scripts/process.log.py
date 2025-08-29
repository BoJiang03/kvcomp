import re

from utils.config import KVCompCacheConfig
from utils.serialization import save

# Read the log file
with open('whole_log.log', 'r') as file:
    log_data = file.read()

# Define the regex pattern to match the three-line data
pattern = re.compile(
    r"(\d+)\n"
    r"(\{.*?\})\n"
    r"(\{.*?\})",
    re.DOTALL
)

# Find all matches
matches = pattern.findall(log_data)

setting_map = {}
accuracy_result_map = {}

# Process the extracted data
for match in matches:
    hash_value = match[0]

    # Clean the result string before evaluation
    data_str = match[2].replace("np.float64", "float")

    # Safely evaluate the result string
    accuracy_res = eval(data_str+"}")

    # Extract the only key from the result map
    result_key = next(iter(accuracy_res.keys()))
    setting_map[hash_value] = (result_key, KVCompCacheConfig.from_str(match[1]))
    accuracy_result_map[hash_value] = accuracy_res

save(setting_map, "./data/recovered_setting_map_llama2_13b_phi4_mistral.pkl")
save(accuracy_result_map, "./data/recovered_accuracy_result_map_llama2_13b_phi4_mistral.pkl")