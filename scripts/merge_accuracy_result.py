from utils.serialization import load, save

accuracy_result_paths = ["./data/gsm8k_coqa_accuracy_result_map_1.pkl", "./data/gsm8k_coqa_accuracy_result_map_2.pkl"]
merge_to_ = "./data/kv_combine_accuracy_result_map.pkl"
accuracy_result_map = {}

for accuracy_result_path in accuracy_result_paths:
    accuracy_result_map_ = load(accuracy_result_path)
    for key, value in accuracy_result_map_.items():
        if key not in accuracy_result_map:
            accuracy_result_map[key] = value
        else:
            print(f"Key {key} already exists in the merged map. Skipping.")
            continue
    print(f"Loaded {len(accuracy_result_map_)} items from {accuracy_result_path}.")

print(f"Total items in merged map: {len(accuracy_result_map)}.")
save(accuracy_result_map, merge_to_)