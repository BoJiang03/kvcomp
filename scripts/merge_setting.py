from utils.serialization import load, save

setting_paths = ["./data/gsm8k_coqa_setting_map_1.pkl", "./data/gsm8k_coqa_setting_map_2.pkl"]
merge_to_ = "./data/kv_combine_setting_map.pkl"
setting_map = {}

for setting_path in setting_paths:
    setting_map_ = load(setting_path)
    for key, value in setting_map_.items():
        if key not in setting_map:
            setting_map[key] = value
        else:
            print(f"Key {key} already exists in the merged map. Skipping.")
            continue
    print(f"Loaded {len(setting_map_)} items from {setting_path}.")

print(f"Total items in merged map: {len(setting_map)}.")
save(setting_map, merge_to_)