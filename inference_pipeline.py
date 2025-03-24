#!/usr/bin/env python3
import os
import sys
import argparse
import tarfile
import shutil
import multiprocessing
import yaml
import torch
import numpy as np
from datetime import date
from typing import List, Dict, Any

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


def data_loader_worker(tar_paths: List[str], temp_dir: str, 
                      data_queue: multiprocessing.Queue, halt_on_error: bool) -> None:
    """
    Worker function to load data from tar files and put the extracted paths into the queue.
    
    Args:
        tar_paths: List of tar file paths to process
        temp_dir: Temporary directory for extraction
        data_queue: Queue to put extracted directory paths
        halt_on_error: Whether to halt on error
    """
    for tar_path in tar_paths:
        try:
            extracted_dir = load_data(tar_path, temp_dir)
            # Put both the extracted directory and the source tar path in the queue
            data_queue.put((tar_path, extracted_dir, None))
        except Exception as e:
            error_msg = f"Error loading data from {tar_path}: {str(e)}"
            print(error_msg)
            if halt_on_error:
                # Signal error to processing workers
                data_queue.put((tar_path, None, error_msg))
                break
            else:
                # Continue with other files but mark this one as failed
                data_queue.put((tar_path, None, error_msg))
    
    # Signal that all data has been loaded
    data_queue.put((None, None, None))


def processor_worker(data_queue: multiprocessing.Queue, results_queue: multiprocessing.Queue, 
                    checkpoint_path: str, config_path: str, output_folder: str, 
                    processing_args: Dict[str, Any], halt_on_error: bool) -> None:
    """
    Worker function to process data from the queue.
    
    Args:
        data_queue: Queue containing extracted directory paths
        results_queue: Queue to store processing results
        checkpoint_path: Path to model checkpoint file
        config_path: Path to configuration file
        output_folder: Folder to save the output
        processing_args: Additional arguments for processing
        halt_on_error: Whether to halt on error
    """
    # Load model inside the process
    model, _ = utils.load_model_from_configs_and_checkpoint(config_path, config_path, checkpoint_path)
    model.to("cuda")
    model.eval()
    model = torch.compile(model)
    print(f"Model loaded successfully in processor {multiprocessing.current_process().name}")
    
    while True:
        tar_path, extracted_dir, error = data_queue.get()
        
        # Check if this is the termination signal
        if tar_path is None:
            # Put the termination signal back for other workers
            data_queue.put((None, None, None))
            break
        
        # If there was an error during loading
        if extracted_dir is None:
            results_queue.put((tar_path, False))
            if halt_on_error:
                break
            continue
        
        try:
            # Process the data
            success = process_data(extracted_dir, model, output_folder, processing_args)
            results_queue.put((tar_path, success))
            
            # Clean up regardless of processing result
            cleanup(extracted_dir)
            
            if not success and halt_on_error:
                break
                
        except Exception as e:
            error_msg = f"Error processing {tar_path}: {str(e)}"
            print(error_msg)
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
    parser.add_argument('--output-folder', '-o', dest='output_folder', type=str,
                        help='Output folder for results')
    parser.add_argument('--checkpoint', dest='checkpoint', type=str,
                        help='Path to model checkpoint file')
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
        
    if not args.tar_files:
        print("Error: No tar files specified.")
        sys.exit(1)
        
    if not args.output_folder:
        print("Error: No output folder specified.")
        sys.exit(1)
        
    if not args.checkpoint or not os.path.exists(args.checkpoint):
        print("Error: Valid checkpoint file must be specified.")
        sys.exit(1)
        
    if not args.config or not os.path.exists(args.config):
        print("Error: Valid config file must be specified.")
        sys.exit(1)
        
    # Verify that all tar files exist
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
    
    # Load model
    # print("Loading model from checkpoint...")
    # model, _ = utils.load_model_from_configs_and_checkpoint(args.config, args.config, args.checkpoint)
    # model.to("cuda")
    # model.eval()
    # model = torch.compile(model)
    # print("Model loaded successfully.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)
    
    # Use the provided tar files directly
    all_tar_files = args.tar_files
    print(f"Processing {len(all_tar_files)} tar files")
    
    # Check if distributed processing is requested
    if args.world_size > 1:
        print(f"Running as distributed task {args.rank} of {args.world_size}")
        tar_files = get_slurm_task_files(all_tar_files, args.world_size, args.rank)
    else:
        tar_files = all_tar_files

    N = len(tar_files)

    if not args.overwrite:
        tarfilenames = [os.path.splitext(os.path.basename(f))[0] for f in tar_files]
        output_filepaths = [os.path.join(tfn[:-5]) + ".tif" for tfn in tarfilenames]
        tar_files = [tf for (tf, of) in zip(tar_files, output_filepaths) if not os.path.exists(of)]

    print(f"This task will process {len(tar_files)} files. {N - len(tar_files)} files will be skipped.")

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
    
    # Divide tar files among loader processes
    num_loaders = max(min(args.parallel_loaders, len(tar_files)), 1)
    files_per_loader = len(tar_files) // num_loaders if num_loaders > 0 else 0
    remainder = len(tar_files) % num_loaders if num_loaders > 0 else 0
    
    loader_processes = []
    start_idx = 0
    
    for i in range(num_loaders):
        # Calculate the slice of tar files for this loader
        end_idx = start_idx + files_per_loader
        if i < remainder:
            end_idx += 1
        
        # Create and start the loader process
        loader_process = multiprocessing.Process(
            target=data_loader_worker,
            args=(tar_files[start_idx:end_idx], args.temp_dir, data_queue, not args.continue_on_error),
            name=f"Loader-{i}"
        )
        loader_process.start()
        loader_processes.append(loader_process)
        
        start_idx = end_idx
    
    # Create and start processor processes
    processor_processes = []
    for i in range(args.parallel_processors):
        processor_process = multiprocessing.Process(
            target=processor_worker,
            args=(data_queue, results_queue, args.checkpoint, args.config, 
                  args.output_folder, processing_args, not args.continue_on_error),
            name=f"Processor-{i}"
        )
        processor_process.start()
        processor_processes.append(processor_process)
    
    # Wait for loader processes to complete
    for process in loader_processes:
        process.join()
    
    # Collect results while processors are still running
    results = {}
    completed_processors = 0
    total_processors = args.parallel_processors
    
    while completed_processors < total_processors:
        if not any(p.is_alive() for p in processor_processes):
            # All processors have exited
            break
            
        try:
            # Get results with a timeout to periodically check if processes are still alive
            tar_path, success = results_queue.get(timeout=0.1)
            results[tar_path] = success
        except:
            # Queue.Empty or other exceptions
            continue
    
    # Drain any remaining results from the queue
    while not results_queue.empty():
        try:
            tar_path, success = results_queue.get(timeout=0.1)
            results[tar_path] = success
        except:
            break
    
    # Make sure all processes have terminated
    for process in processor_processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=2)
            if process.is_alive():
                print(f"Warning: Process {process.name} did not terminate cleanly")
    
    # Report results
    successes = sum(1 for success in results.values() if success)
    failures = sum(1 for success in results.values() if not success)
    print(f"Processing complete: {successes} succeeded, {failures} failed")
    
    # List failed files if any
    if failures > 0:
        print("Failed files:")
        for tar_path, success in results.items():
            if not success:
                print(f"  {tar_path}")
        
        # Return non-zero exit code if there were failures
        if not args.continue_on_error:
            sys.exit(1)

#%%
if __name__ == "__main__":
    main()

#%%

