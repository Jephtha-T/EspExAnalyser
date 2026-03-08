import os

# Import subsystems
from Frame_Extraction import extract_frames
from Portafilter_Tracking import process_portafilter_tracking
from Feature_Extraction import extract_features_from_video
from Data_Export import export_to_csv
try:
    from Espresso_Model import predict_from_results_json
except Exception:
    predict_from_results_json = None

base_dir = os.path.dirname(os.path.abspath(__file__))
video_dir = os.path.join(base_dir, "Video Data")
image_dir = os.path.join(base_dir, "Image Data")
frames_dir = os.path.join(base_dir, "Image Data", "Frames")
cropped_dir = os.path.join(base_dir, "Image Data", "Cropped")
analysis_dir = os.path.join(base_dir, "Analysis")
analysis_blond_dir = os.path.join(analysis_dir, "blond")
analysis_channeling_dir = os.path.join(analysis_dir, "channeling")
analysis_results_dir = os.path.join(analysis_dir, "results")
video_name = "test1.mp4"
run_mode = True
export_mode = False
predict_mode = False
model_path = os.path.join(analysis_dir, "extraction_model.joblib")

# Create necessary directories
for directory in [
    image_dir,
    frames_dir,
    cropped_dir,
    analysis_dir,
    analysis_blond_dir,
    analysis_channeling_dir,
    analysis_results_dir,
]:
    os.makedirs(directory, exist_ok=True)


def get_video_name(video_path):
    # Keep output names consistent with the source video file.
    return os.path.basename(video_path)


def try_predict_extraction_level(results_json_path, model_path=model_path):
    if predict_from_results_json is None:
        print("Warning: Espresso_Model module unavailable. Prediction skipped.")
        return None
    if not os.path.exists(model_path):
        print(f"Warning: model file not found: {model_path}. Prediction skipped.")
        return None
    if not os.path.exists(results_json_path):
        print(f"Warning: results JSON not found: {results_json_path}. Prediction skipped.")
        return None

    try:
        prediction = predict_from_results_json(results_json_path, model_path=model_path)
        print("\n[MODEL] EXTRACTION PREDICTION")
        print(f"  - Predicted class: {prediction['predicted_class']}")
        print(f"  - Predicted label: {prediction['predicted_label']}")
        if prediction.get("confidence") is not None:
            print(f"  - Confidence: {prediction['confidence']:.4f}")
        return prediction
    except Exception as e:
        print(f"Warning: prediction failed: {e}")
        return None


def clear_directory(directory):
    # Remove files and folders recursively.
    if os.path.exists(directory):
        for fname in os.listdir(directory):
            fpath = os.path.join(directory, fname)
            try:
                if os.path.isfile(fpath):
                    os.unlink(fpath)
                elif os.path.isdir(fpath):
                    clear_directory(fpath)
                    os.rmdir(fpath)
            except Exception as e:
                print(f"Warning: could not delete {fpath}: {e}")

def clear_image_data():
    # Reset image artifacts from previous runs.
    print(f"\nClearing image data in {image_dir}...")
    clear_directory(image_dir)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(cropped_dir, exist_ok=True)


def get_video_paths(mode=True, video_name=video_name):
    if mode == True:
        return [os.path.join(video_dir, video_name)]

    else:
        valid_exts = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".wmv")
        videos = []
        if os.path.isdir(video_dir):
            for fname in sorted(os.listdir(video_dir)):
                fpath = os.path.join(video_dir, fname)
                if os.path.isfile(fpath) and fname.lower().endswith(valid_exts):
                    videos.append(fpath)
        return videos


def run_espresso_analysis_pipeline(video_path, 
                                   subsystems_to_run=['frame_extraction', 'portafilter_tracking', 'feature_extraction'],
                                   show_gui=True):
    # Run the full analysis pipeline for one video.
    
    # Validate input
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    clear_image_data()

    video_name = get_video_name(video_path)
    print("\n" + "="*60)
    print(f"ESPRESSO ANALYSIS PIPELINE")
    print(f"Video: {video_name}")
    print("="*60)
    
    results = {
        "video": video_name,
        "video_path": video_path,
        "stages": {}
    }
    
    # Stage 1: Frame Extraction
    if 'frame_extraction' in subsystems_to_run:
        print("\n[STAGE 1/3] FRAME EXTRACTION")
        print("-" * 60)
        try:
            frame_count = extract_frames(video_path, frames_dir, target_fps=1)
            results["stages"]["frame_extraction"] = {
                "status": "success",
                "frame_count": frame_count,
                "output_directory": frames_dir
            }
            print(f"Frame extraction successful: {frame_count} frames")
        except Exception as e:
            print(f"Frame extraction failed: {e}")
            results["stages"]["frame_extraction"] = {
                "status": "failed",
                "error": str(e)
            }
            return results
    
    # Stage 2: Portafilter Tracking (Crop + Stabilise)
    if 'portafilter_tracking' in subsystems_to_run:
        print("\n[STAGE 2/3] PORTAFILTER DETECTION, CROPPING & STABILISATION")
        print("-" * 60)
        try:
            tracking_result = process_portafilter_tracking(
                frames_dir=frames_dir,
                output_dir=cropped_dir
            )
            results["stages"]["portafilter_tracking"] = {
                "status": "success",
                "ellipse": str(tracking_result["ellipse"]),
                "hole_mode_size": tracking_result["hole_mode_size"],
                "frame_count": tracking_result["frame_count"],
                "output_directory": cropped_dir
            }
            print(f"Portafilter tracking successful: {tracking_result['frame_count']} cropped frames")
        except Exception as e:
            print(f"Portafilter tracking failed: {e}")
            results["stages"]["portafilter_tracking"] = {
                "status": "failed",
                "error": str(e)
            }
            return results
    
    # Stage 3: Feature Extraction
    if 'feature_extraction' in subsystems_to_run:
        print("\n[STAGE 3/3] FEATURE EXTRACTION (BLONDING & CHANNELING)")
        print("-" * 60)
        try:
            # Strip file extension from video_name for output files
            video_name_no_ext = os.path.splitext(video_name)[0]
            
            feature_results = extract_features_from_video(
                cropped_frames_dir=cropped_dir,
                video_name=video_name_no_ext,
                output_dir=analysis_dir,
                output_blond_dir=analysis_blond_dir,
                output_channeling_dir=analysis_channeling_dir,
                output_results_dir=analysis_results_dir,
                save_plots=True,
                detect_channeling=True,
                show_gui=show_gui
            )
            
            blond_file = os.path.join(analysis_blond_dir, f"{video_name_no_ext}_Blond.png")
            channeling_file = os.path.join(analysis_channeling_dir, f"{video_name_no_ext}_Channeling.png")
            results_file = os.path.join(analysis_results_dir, f"{video_name_no_ext}_results.json")
            
            results["stages"]["feature_extraction"] = {
                "status": "success",
                "blond_frame": feature_results["blond_frame"] if feature_results else None,
                "blond_rate": feature_results["blond_rate"] if feature_results else None,
                "output_directory": analysis_dir,
                "output_directories": {
                    "blond": analysis_blond_dir,
                    "channeling": analysis_channeling_dir,
                    "results": analysis_results_dir,
                },
                "output_files": {
                    "blond_plot": blond_file if os.path.exists(blond_file) else None,
                    "channeling_plot": channeling_file if os.path.exists(channeling_file) else None,
                    "results_json": results_file if os.path.exists(results_file) else None
                }
            }
            print(f"Feature extraction successful")
            if feature_results:
                print(f"  - Blonding frame: {feature_results['blond_frame']}")
                print(f"  - Blonding rate: {feature_results['blond_rate']:.4f}")

            if predict_mode:
                prediction = try_predict_extraction_level(results_file, model_path=model_path)
                if prediction is not None:
                    results["stages"]["prediction"] = {
                        "status": "success",
                        "predicted_class": prediction["predicted_class"],
                        "predicted_label": prediction["predicted_label"],
                        "confidence": prediction.get("confidence"),
                        "model_path": model_path,
                    }
                else:
                    results["stages"]["prediction"] = {
                        "status": "skipped",
                        "model_path": model_path,
                    }
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            results["stages"]["feature_extraction"] = {
                "status": "failed",
                "error": str(e)
            }
            return results
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    for stage, stage_result in results["stages"].items():
        status_icon = "V" if stage_result.get("status") == "success" else "X"
        print(f"{status_icon} {stage}: {stage_result.get('status')}")
    
    # Save pipeline results
    # results_file = os.path.join(analysis_dir, f"{video_name}_pipeline.json")
    # results_to_save = results.copy()
    # results_to_save["video_path"] = str(results_to_save["video_path"])
    # with open(results_file, "w") as f:
    # print(f"\nPipeline results saved to {results_file}")
    
    return results


def main():
    # Entry point for export mode or full analysis mode.
    
    # If export mode is enabled, only build the CSV.
    if export_mode:
        export_to_csv(analysis_dir, output_file="training_data.csv")
        return

    video_paths = get_video_paths(mode=run_mode, video_name=video_name)
    if not video_paths:
        if not run_mode:
            print(f"No video files found in {video_dir}")
        else:
            print(f"Video not found: {os.path.join(video_dir, video_name)}")
        return

    mode_str = f"single video ({video_name})" if run_mode else "all videos"
    print(f"Mode: {mode_str}")
    print(f"Videos queued: {len(video_paths)}")

    all_results = {}
    for idx, video_path in enumerate(video_paths, start=1):
        print(f"\nStarting run {idx}/{len(video_paths)}")
        all_results[os.path.basename(video_path)] = run_espresso_analysis_pipeline(video_path, show_gui=run_mode)

    print("\n" + "="*60)
    print("ALL REQUESTED RUNS COMPLETE")
    print("="*60)
    success_count = 0
    failed_count = 0
    for video_file, result in all_results.items():
        stage_failures = [
            stage for stage, stage_result in result.get("stages", {}).items()
            if stage_result.get("status") != "success"
        ]
        status = "success" if not stage_failures else "failed"
        if status == "success":
            success_count += 1
        else:
            failed_count += 1
        print(f"{video_file}: {status}")

    print(f"\nBatch summary: {success_count} success, {failed_count} failed")

if __name__ == "__main__":
    main()

