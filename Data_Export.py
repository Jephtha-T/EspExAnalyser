# Export analysis JSON files to a training CSV.

import os
import json
import csv


def load_all_results(analysis_dir):
    # Load all *_results.json files from Analysis or Analysis/results.
    results_list = []
    
    if not os.path.exists(analysis_dir):
        print(f"Warning: analysis directory not found: {analysis_dir}")
        return results_list
    
    results_subdir = os.path.join(analysis_dir, "results")
    search_dirs = [results_subdir, analysis_dir] if os.path.isdir(results_subdir) else [analysis_dir]

    loaded_paths = set()
    for current_dir in search_dirs:
        for filename in sorted(os.listdir(current_dir)):
            if not filename.endswith("_results.json"):
                continue

            filepath = os.path.join(current_dir, filename)
            if filepath in loaded_paths:
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    results_list.append(data)
                    loaded_paths.add(filepath)
            except Exception as e:
                print(f"Warning: could not load {filename}: {e}")
    
    return results_list


def normalize_value(value, min_val, max_val):
    # Min-max normalization to [0, 1].
    if max_val == min_val:
        return 0.5  # Avoid division by zero
    return (value - min_val) / (max_val - min_val)


def label_to_class_id(label):
    # Map label text to class id.
    if not label:
        return None

    normalized = str(label).strip().lower()

    if 'under' in normalized:
        return 0
    if 'ideal' in normalized:
        return 1
    if 'over' in normalized:
        return 2

    return None


def load_labels(labels_csv_path):
    # Load labels from Video Data/labels.csv.
    labels_by_video = {}
    labels_in_order = []

    if not os.path.exists(labels_csv_path):
        print(f"Warning: labels file not found: {labels_csv_path}")
        return labels_by_video, labels_in_order

    try:
        parse_error = None
        for encoding in ('utf-8-sig', 'cp1252', 'latin-1'):
            try:
                labels_by_video.clear()
                labels_in_order.clear()

                with open(labels_csv_path, 'r', encoding=encoding, newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        label_text = (row.get('Espresso Extraction Level') or '').strip()
                        label_id = label_to_class_id(label_text)
                        if label_id is None:
                            continue

                        labels_in_order.append(label_id)

                        row_id = (row.get('Id') or row.get('id') or '').strip()
                        if row_id:
                            try:
                                labels_by_video[f"test{int(float(row_id))}"] = label_id
                            except ValueError:
                                pass
                parse_error = None
                break
            except UnicodeDecodeError as e:
                parse_error = e
                continue

        if parse_error is not None:
            raise parse_error

        print(f"Loaded {len(labels_in_order)} labels from labels.csv")
    except Exception as e:
        print(f"Warning: could not load labels.csv: {e}")

    return labels_by_video, labels_in_order


def export_to_csv(analysis_dir, output_file="training_data.csv"):
    # Build training_data.csv from extracted feature JSON files.
    print("\n" + "="*60)
    print("EXPORTING DATA TO CSV FOR AI MODEL")
    print("="*60)
    
    # Load all analysis results
    results_list = load_all_results(analysis_dir)
    
    if not results_list:
        print("No results files found. Run analysis first.")
        return None
    
    print(f"Loaded {len(results_list)} result files")

    base_dir = os.path.dirname(analysis_dir)
    labels_csv_path = os.path.join(base_dir, "Video Data", "labels.csv")
    labels_by_video, labels_in_order = load_labels(labels_csv_path)
    
    # Extract raw features and find min/max for normalization
    raw_data = []
    for result in results_list:
        video_name = result.get('video_name', 'unknown')
        
        # Calculate shot time (in frames)
        flow_start = result.get('flow_start', 0)
        flow_end = result.get('flow_end', 0)
        shot_time = flow_end - flow_start
        
        # Extract features
        blonding_rate = result.get('blond_rate', 0.0)
        channeling_stats = result.get('channeling_stats', {})
        channeling_avg = channeling_stats.get('average', 0.0)
        channeling_max = channeling_stats.get('max', 0.0)
        channeling_min = channeling_stats.get('min', 0.0)
        
        raw_data.append({
            'video_id': video_name,
            'shot_time': shot_time,
            'blonding_rate': blonding_rate,
            'channeling_avg': channeling_avg,
            'channeling_max': channeling_max,
            'channeling_min': channeling_min
        })
    
    # Calculate min/max for non-channeling features for normalization
    shot_times = [d['shot_time'] for d in raw_data]
    blonding_rates = [d['blonding_rate'] for d in raw_data]
    
    min_shot_time, max_shot_time = min(shot_times), max(shot_times)
    min_blonding, max_blonding = min(blonding_rates), max(blonding_rates)
    
    print(f"\nFeature ranges for normalization:")
    print(f"  Shot time: {min_shot_time} - {max_shot_time} frames")
    print(f"  Blonding rate: {min_blonding:.4f} - {max_blonding:.4f}")
    print(f"  Channeling: normalized per-shot (using each shot's own min/max)")
    
    # Prepare CSV data with normalized values
    csv_data = []
    for index, data in enumerate(raw_data):
        # Calculate per-shot channeling metrics
        ch_min = data['channeling_min']
        ch_max = data['channeling_max']
        ch_avg = data['channeling_avg']
        
        # Range shows variation in hole count (higher = more inconsistent coverage)
        ch_range = ch_max - ch_min
        ch_range_norm = ch_range / ch_max if ch_max > 0 else 0 # normalized range (0-1) for channeling variation
        
        # Coverage position: where avg sits between min and max (0=min, 0.5=middle, 1=max)
        # Shows if channeling tends toward min or max during the shot
        if ch_range > 0:
            ch_coverage_norm = (ch_avg - ch_min) / ch_range
            ch_coverage_norm = max(0.0, min(1.0, ch_coverage_norm))  # Clamp to [0, 1]
        else:
            ch_coverage_norm = 0.5  # No variation, perfectly consistent
        
        label = labels_by_video.get(data['video_id'])
        if label is None and index < len(labels_in_order):
            label = labels_in_order[index]
        if label is None:
            label = ''

        row = {
            'video_id': data['video_id'],
            'label': label,
            'shot_time': data['shot_time'],
            'blonding_rate': data['blonding_rate'],
            'blonding_rate_norm': normalize_value(data['blonding_rate'], min_blonding, max_blonding),
            'channeling_range': ch_range,
            'channeling_range_norm': ch_range_norm,
            'channeling_coverage_norm': ch_coverage_norm
        }
        csv_data.append(row)
    
    # Write to CSV
    output_path = os.path.join(analysis_dir, output_file)
    fieldnames = [
        'video_id', 
        'label',
        'shot_time',
        'blonding_rate', 'blonding_rate_norm',
        'channeling_range', 'channeling_range_norm', 'channeling_coverage_norm'
    ]
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_data)
        
        print(f"\nTraining data exported to: {output_path}")
        print(f"Total rows: {len(csv_data)}")
        print(f"Features: {len(fieldnames)}")
        
        # Display sample data
        print(f"\nSample data (first 3 rows):")
        for i, row in enumerate(csv_data[:3], 1):
            print(f"  Row {i}: {row['video_id']}")
            print(f"    Blonding rate: {row['blonding_rate']:.4f} (normalized: {row['blonding_rate_norm']:.4f})")
            print(f"    Channeling range: {row['channeling_range']:.2f} holes")
            print(f"    Channeling range normalized: {row['channeling_range_norm']:.4f}")
            print(f"    Channeling coverage: {row['channeling_coverage_norm']:.4f} (0=min, 0.5=middle, 1=max)")
        
        print("\n" + "="*60)
        return output_path
        
    except Exception as e:
        print(f"Could not write CSV: {e}")
        return None


if __name__ == "__main__":
    # Example usage.
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    analysis_dir = os.path.join(base_dir, "Analysis")

    export_to_csv(analysis_dir, output_file="training_data.csv")
