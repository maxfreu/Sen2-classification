#!/usr/bin/env python3
import os
import sys
import argparse
import tarfile
import shutil
import subprocess
import multiprocessing
import yaml
import torch
import numpy as np
from datetime import date
from typing import List, Dict, Any, Optional

from sen2classification import utils


def load_data(tar_path: str, temp_dir: str) -> str:
    """
    Extract a tar file to a temporary directory.
    
    Args:
        tar_path: Path to the tar file
        temp_dir: Path to the temporary directory
        
    Returns:
        Path to the extracted directory
    """
    # Create a subfolder based on the tar filename
    tile_name = os.path.basename(os.path.dirname(tar_path))
    extract_dir = os.path.join(temp_dir, tile_name)
    
    # Create the extraction directory if it doesn't exist
    os.makedirs(extract_dir, exist_ok=True)
    
    # Extract the tar file
    print(f"Extracting {tar_path} to {extract_dir}")
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(path=extract_dir)
    
    return extract_dir


def process_data(extracted_dir: str, model, output_folder: str, args: Dict[str, Any]) -> bool:
    """
    Process the data in the extracted directory using the ML model.
    
    Args:
        extracted_dir: Path to the directory containing extracted data
        model: The PyTorch model for inference
        output_folder: Folder to save the output
        args: Additional arguments for processing
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        print(f"Processing data in {extracted_dir}")
        
        # Generate output filepath
        tile_name = os.path.basename(extracted_dir)
        output_filepath = os.path.join(output_folder, f"{tile_name}.tif")
        
        # Convert date strings to date objects if needed
        tmin_data = args.get('tmin_data')
        tmax_data = args.get('tmax_data')
        tmin_inference = args.get('tmin_inference', tmin_data)
        tmax_inference = args.get('tmax_inference', tmax_data)
        
        # Get other parameters from args
        sequence_length = args.get('sequence_length', 64)
        qai = args.get('qai', 31)
        apply_argmax = not args.get('soft', False)
        num_classes = args.get('num_classes', 0)
        mean = args.get('mean')
        stddev = args.get('stddev')
        time_encoding = args.get('time_encoding', 'doy')
        append_ndvi = args.get('append_ndvi', False)
        batch_size = args.get('batch_size', 3000)
        
        # Call the model's predict method
        model.predict_force_folder(
            extracted_dir,
            seq_len=sequence_length,
            qai=qai,
            output_filepath=output_filepath,
            verbose=False,
            time_encoding=time_encoding,
            mean=mean,
            stddev=stddev,
            batch_size=batch_size,
            apply_argmax=apply_argmax,
            num_classes=num_classes,
            band_reordering=(3,0,1,2,4,5,6,7,8,9,10,11,12,13),
            tmin_data=tmin_data,
            tmax_data=tmax_data,
            tmin_inference=tmin_inference,
            tmax_inference=tmax_inference,
            append_ndvi=append_ndvi
        )
        
        print(f"Successfully processed {extracted_dir} and saved to {output_filepath}")
        return True
    
    except Exception as e:
        print(f"Error during processing of {extracted_dir}: {str(e)}")
        return False


def cleanup(extracted_dir: str) -> None:
    """
    Clean up the temporary directory.
    
    Args:
        extracted_dir: Path to the directory to clean up
    """
    print(f"Cleaning up {extracted_dir}")
    if os.path.exists(extracted_dir):
        shutil.rmtree(extracted_dir)


def load_data_s3(s3_prefix: str, tile_name: str, temp_dir: str) -> str:
    """
    Download a tile folder from S3 to a temporary directory using rclone.
    
    Args:
        s3_prefix: S3 prefix, e.g. 's3-force:forst-sentinel2/force/L2/ard'
        tile_name: Tile folder name, e.g. 'X0072_Y0049'
        temp_dir: Path to the temporary directory
        
    Returns:
        Path to the downloaded directory
    """
    s3_path = f"{s3_prefix}/{tile_name}/"
    local_dir = os.path.join(temp_dir, tile_name)
    os.makedirs(local_dir, exist_ok=True)
    
    print(f"Downloading {s3_path} to {local_dir}")
    result = subprocess.run(
        ["rclone", "copy", s3_path, local_dir, "--transfers=8"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"rclone copy failed: {result.stderr}")
    
    return local_dir


def data_loader_worker(items: List[str], temp_dir: str, 
                      data_queue: multiprocessing.Queue, halt_on_error: bool,
                      s3_prefix: Optional[str] = None) -> None:
    """
    Worker function to load data (from tar files or S3) and put the extracted paths into the queue.
    
    Args:
        items: List of tar file paths or tile names to process
        temp_dir: Temporary directory for extraction
        data_queue: Queue to put extracted directory paths
        halt_on_error: Whether to halt on error
        s3_prefix: If set, items are tile names to download from this S3 prefix
    """
    for item in items:
        try:
            if s3_prefix:
                extracted_dir = load_data_s3(s3_prefix, item, temp_dir)
            else:
                extracted_dir = load_data(item, temp_dir)
            data_queue.put((item, extracted_dir, None))
        except Exception as e:
            error_msg = f"Error loading data from {item}: {str(e)}"
            print(error_msg)
            if halt_on_error:
                data_queue.put((item, None, error_msg))
                break
            else:
                data_queue.put((item, None, error_msg))
    
    # Signal that all data has been loaded
    data_queue.put((None, None, None))


def processor_worker(data_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue, 
                    checkpoint_paths: List[str], config_path: str, output_folders: List[str], 
                    processing_args: Dict[str, Any], halt_on_error: bool) -> None:
    """
    Worker function to process data from the queue.
    
    Args:
        data_queue: Queue containing extracted directory paths
        results_queue: Queue to store processing results
        checkpoint_paths: List of paths to model checkpoint files
        config_path: Path to configuration file
        output_folders: List of folders to save the outputs (one per model)
        processing_args: Additional arguments for processing
        halt_on_error: Whether to halt on error
    """
    # Load multiple models inside the process
    models = []
    for i, checkpoint_path in enumerate(checkpoint_paths):
        model, _ = utils.load_model_from_configs_and_checkpoint(config_path, config_path, checkpoint_path)
        model.to("cuda")
        model.eval()
        model = torch.compile(model)
        models.append(model)
        print(f"Model {i+1}/{len(checkpoint_paths)} loaded successfully in processor {multiprocessing.current_process().name}")
    
    while True:
        tar_path, extracted_dir, error = data_queue.get()
        
        # Check if this is the termination signal
        if tar_path is None:
            # Put the termination signal back for other workers
            data_queue.put((None, None, None))
            break
        
        # If there was an error during loading
        if extracted_dir is None:
            # Report failure for all models
            for _ in range(len(models)):
                results_queue.put((tar_path, False))
            if halt_on_error:
                break
            continue
        
        try:
            # Process the data with each model
            successes = []
            for i, model in enumerate(models):
                success = process_data(extracted_dir, model, output_folders[i], processing_args)
                successes.append(success)
                results_queue.put((tar_path, success))
                
                if not success and halt_on_error:
                    break
            
            # Clean up regardless of processing results
            cleanup(extracted_dir)
            
            if not all(successes) and halt_on_error:
                break
                
        except Exception as e:
            error_msg = f"Error processing {tar_path}: {str(e)}"
            print(error_msg)
            # Report failure for all models
            for _ in range(len(models)):
                results_queue.put((tar_path, False))
            
            # Attempt cleanup even if processing failed
            try:
                cleanup(extracted_dir)
            except Exception as cleanup_error:
                print(f"Error during cleanup of {extracted_dir}: {str(cleanup_error)}")
            
            if halt_on_error:
                break


def get_slurm_task_files(all_files: List[str], num_tasks: int, task_id: int) -> List[str]:
    """
    Get the subset of files to process for this SLURM task.
    
    Args:
        all_files: List of all files to process
        num_tasks: Total number of SLURM tasks
        task_id: Current SLURM task ID
        
    Returns:
        List of files to process for this task
    """
    files_per_task = len(all_files) // num_tasks
    remainder = len(all_files) % num_tasks
    
    start_idx = task_id * files_per_task + min(task_id, remainder)
    end_idx = start_idx + files_per_task + (1 if task_id < remainder else 0)
    
    return all_files[start_idx:end_idx]


def main():
    parser = argparse.ArgumentParser(description='Process satellite image tiles in parallel.')
    parser.add_argument('--tar-files', '-i', dest='tar_files', nargs='+', type=str,
                        help='List of tar files to process')
    parser.add_argument('--s3-prefix', dest='s3_prefix', type=str, default=None,
                        help='S3 rclone prefix for tile folders, e.g. s3-force:forst-sentinel2/force/L2/ard')
    parser.add_argument('--tiles', dest='tiles', nargs='+', type=str,
                        help='List of tile names to download from S3 (used with --s3-prefix)')
    parser.add_argument('--output-folder', '-o', dest='output_folder', type=str,
                        help='Base output folder for results')
    parser.add_argument('--checkpoints', dest='checkpoints', nargs='+', type=str,
                        help='List of paths to model checkpoint files')
    parser.add_argument('--config', dest='config', type=str,
                        help='Path to configuration file')
    parser.add_argument('--qai', dest='qai', type=int, default=31,
                        help='Quality Assessment Index threshold')
    parser.add_argument('--sequence-length', dest='sequence_length', type=int, default=64,
                        help='Sequence length for the model')
    parser.add_argument('--tmin-data', dest='tmin_data', type=str,
                        help='Starting time for loading data. Format: yyyy-mm-dd')
    parser.add_argument('--tmax-data', dest='tmax_data', type=str,
                        help='Ending time for loading data (exclusive). Format: yyyy-mm-dd')
    parser.add_argument('--tmin-inference', dest='tmin_inference', type=str, default='',
                        help='Starting time for inference. Format: yyyy-mm-dd. Default: tmin_data')
    parser.add_argument('--tmax-inference', dest='tmax_inference', type=str, default='',
                        help='Ending time for inference (exclusive). Format: yyyy-mm-dd. Default: tmax_data')
    parser.add_argument('--soft', action='store_true',
                        help='If given, the output will be a multiband raster where each band represents the probability '
                             'for a given class, scaled to 0-255.')
    parser.add_argument('--num-classes', dest='num_classes', default=0, type=int,
                        help='Number of classes the network outputs. Only required if --soft is given.')
    parser.add_argument('--temp-dir', dest='temp_dir', type=str,
                        default=os.environ.get('LOCAL_TMPDIR', '/tmp'),
                        help='Temporary directory for data extraction')
    parser.add_argument('--parallel-loaders', dest='parallel_loaders', type=int, default=1,
                        help='Number of parallel data loaders')
    parser.add_argument('--parallel-processors', dest='parallel_processors', type=int, default=1,
                        help='Number of parallel data processors')
    parser.add_argument('--queue-size', dest='queue_size', type=int, default=2,
                        help='Maximum size of the queue between loading and processing')
    parser.add_argument('--continue-on-error', action='store_true',
                        help='Continue processing if an error occurs')
    parser.add_argument('--world-size', dest='world_size', type=int, default=1,
                        help='Total number of parallel tasks/processes')
    parser.add_argument('--rank', dest='rank', type=int, default=0,
                        help='Current task/process ID (zero-based)')
    parser.add_argument("--overwrite", help="If given, existing files will be overwritten.",
                        action="store_true")
    parser.add_argument("--batch-size", dest='batch_size', type=int, default=3000)
                        
    args = parser.parse_args()
    
    # Validate arguments
    if args.soft and args.num_classes == 0:
        print("Error: The argument --num-classes must be specified and greater than 0 when --soft is given.")
        sys.exit(1)
        
    # Determine mode: tar files or S3 tiles
    s3_mode = args.s3_prefix is not None
    if s3_mode:
        if not args.tiles:
            print("Error: --tiles must be specified when using --s3-prefix.")
            sys.exit(1)
    else:
        if not args.tar_files:
            print("Error: No tar files specified. Use --tar-files or --s3-prefix with --tiles.")
            sys.exit(1)
        
    if not args.output_folder:
        print("Error: No output folder specified.")
        sys.exit(1)
        
    if not args.checkpoints:
        print("Error: No checkpoint files specified.")
        sys.exit(1)
    
    # Verify that all checkpoint files exist
    for checkpoint_path in args.checkpoints:
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint file does not exist: {checkpoint_path}")
            sys.exit(1)
        
    if not args.config or not os.path.exists(args.config):
        print("Error: Valid config file must be specified.")
        sys.exit(1)
        
    # Verify that all tar files exist (only in tar mode)
    if not s3_mode:
        for tar_file in args.tar_files:
            if not os.path.exists(tar_file):
                print(f"Error: Tar file does not exist: {tar_file}")
                sys.exit(1)
            if not tar_file.endswith('.tar'):
                print(f"Warning: File may not be a tar archive: {tar_file}")
    
    # Parse date ranges
    tmin_data = date(*[int(x) for x in args.tmin_data.split("-")])
    tmax_data = date(*[int(x) for x in args.tmax_data.split("-")])
    tmin_inference = date(*[int(x) for x in args.tmin_inference.split("-")]) if args.tmin_inference else tmin_data
    tmax_inference = date(*[int(x) for x in args.tmax_inference.split("-")]) if args.tmax_inference else tmax_data
    
    # Load configurations
    with open(args.config, "r") as f:
        data_config = yaml.safe_load(f)["data"]
        mean = np.array(data_config["mean"]).astype(np.float32)
        stddev = np.array(data_config["stddev"]).astype(np.float32)
    
    # Create output directories for each model
    output_folders = []
    for checkpoint_path in args.checkpoints:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        checkpoint_id = checkpoint_name.split("_")[-1]
        model_output_folder = os.path.abspath(args.output_folder) + f"_{checkpoint_id}"
        os.makedirs(model_output_folder, exist_ok=True)
        output_folders.append(model_output_folder)
    
    # Build the list of items to process (tar paths or tile names)
    if s3_mode:
        all_items = args.tiles
        print(f"Processing {len(all_items)} S3 tiles using {len(args.checkpoints)} models")
    else:
        all_items = args.tar_files
        print(f"Processing {len(all_items)} tar files using {len(args.checkpoints)} models")
    
    # Check if distributed processing is requested
    if args.world_size > 1:
        print(f"Running as distributed task {args.rank} of {args.world_size}")
        items = get_slurm_task_files(all_items, args.world_size, args.rank)
    else:
        items = all_items

    N = len(items)

    if not args.overwrite:
        # Skip files that already exist in ALL output folders
        filtered_items = []
        for item in items:
            if s3_mode:
                tile_name = item  # tile name directly
            else:
                tarfilename = os.path.splitext(os.path.basename(item))[0]
                tile_name = tarfilename[:-5]  # strip _YEAR suffix
            output_filepaths = [os.path.join(folder, f"{tile_name}.tif") for folder in output_folders]
            if not all(os.path.exists(path) for path in output_filepaths):
                filtered_items.append(item)
        items = filtered_items

    print(f"This task will process {len(items)} files. {N - len(items)} files will be skipped.")

    if len(items) == 0:
        print("No files left to process. Exiting.")
        return

    # Prepare processing arguments
    processing_args = {
        'sequence_length': args.sequence_length,
        'qai': args.qai,
        'soft': args.soft,
        'num_classes': args.num_classes,
        'tmin_data': tmin_data,
        'tmax_data': tmax_data,
        'tmin_inference': tmin_inference,
        'tmax_inference': tmax_inference,
        'mean': mean,
        'stddev': stddev,
        'time_encoding': data_config.get("time_encoding", "doy"),
        'append_ndvi': data_config.get("append_ndvi", False),
        'batch_size': args.batch_size
    }
    
    # Create multiprocessing queues for communication between processes
    multiprocessing.set_start_method('spawn', force=True)  # Use spawn for CUDA compatibility
    data_queue = multiprocessing.Queue(maxsize=args.queue_size)
    results_queue = multiprocessing.Queue()
    
    # Divide items among loader processes
    num_loaders = max(min(args.parallel_loaders, len(items)), 1)
    files_per_loader = len(items) // num_loaders if num_loaders > 0 else 0
    remainder = len(items) % num_loaders if num_loaders > 0 else 0
    
    loader_processes = []
    start_idx = 0
    
    for i in range(num_loaders):
        # Calculate the slice of items for this loader
        end_idx = start_idx + files_per_loader
        if i < remainder:
            end_idx += 1
        
        # Create and start the loader process
        loader_process = multiprocessing.Process(
            target=data_loader_worker,
            args=(items[start_idx:end_idx], args.temp_dir, data_queue, not args.continue_on_error),
            kwargs={'s3_prefix': args.s3_prefix if s3_mode else None},
            name=f"Loader-{i}"
        )
        loader_process.start()
        loader_processes.append(loader_process)
        
        start_idx = end_idx
    
    # Create and start processor processes with all checkpoints and output folders
    processor_processes = []
    for i in range(args.parallel_processors):
        processor_process = multiprocessing.Process(
            target=processor_worker,
            args=(data_queue, results_queue, args.checkpoints, args.config, 
                  output_folders, processing_args, not args.continue_on_error),
            name=f"Processor-{i}"
        )
        processor_process.start()
        processor_processes.append(processor_process)
    
    # Wait for loader processes to complete
    for process in loader_processes:
        process.join()
    
    # Collect results while processors are still running
    results = {checkpoint: {} for checkpoint in args.checkpoints}
    expected_results = len(items) * len(args.checkpoints)
    result_count = 0
    
    while result_count < expected_results:
        if not any(p.is_alive() for p in processor_processes):
            # All processors have exited
            break
            
        try:
            # Get results with a timeout to periodically check if processes are still alive
            tar_path, success = results_queue.get(timeout=0.1)
            # Results from multiple models for the same tar path are sent separately
            # Store them in the results dictionary
            checkpoint_idx = result_count % len(args.checkpoints)
            checkpoint = args.checkpoints[checkpoint_idx]
            results[checkpoint][tar_path] = success
            result_count += 1
        except:
            # Queue.Empty or other exceptions
            continue
    
    # Drain any remaining results from the queue
    while not results_queue.empty():
        try:
            tar_path, success = results_queue.get(timeout=0.1)
            # Assign remaining results
            for checkpoint in args.checkpoints:
                if tar_path not in results[checkpoint]:
                    results[checkpoint][tar_path] = success
                    break
        except:
            break
    
    # Make sure all processes have terminated
    for process in processor_processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                print(f"Warning: Process {process.name} did not terminate cleanly")
    
    # Report results for each model
    for checkpoint_idx, checkpoint in enumerate(args.checkpoints):
        checkpoint_name = os.path.basename(checkpoint)
        model_results = results.get(checkpoint, {})
        successes = sum(1 for success in model_results.values() if success)
        failures = sum(1 for success in model_results.values() if not success)
        print(f"Model {checkpoint_idx+1} ({checkpoint_name}): {successes} succeeded, {failures} failed")
    
    # List failed files if any
    total_failures = sum(1 for checkpoint in args.checkpoints 
                         for success in results.get(checkpoint, {}).values() if not success)
    if total_failures > 0:
        print("Failed files:")
        for checkpoint in args.checkpoints:
            checkpoint_name = os.path.basename(checkpoint)
            model_results = results.get(checkpoint, {})
            for tar_path, success in model_results.items():
                if not success:
                    print(f"  {tar_path} (model: {checkpoint_name})")
        
        # Return non-zero exit code if there were failures
        if not args.continue_on_error:
            sys.exit(1)

#%%
if __name__ == "__main__":
    main()

#%%

