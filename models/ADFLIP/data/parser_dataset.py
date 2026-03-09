import multiprocessing as mp
from data import all_atom_parse as aap
from data.all_atom_parse import dump_structure_data
import glob
import os
from tqdm import tqdm
import argparse

def process_file(raw_data):
    try:
        data = aap.parse_or_load_mmcif(raw_data)
        if 'train' in raw_data:
            save_path = raw_data.replace('train', 'parser_data')
        elif 'test' in raw_data:
            save_path = raw_data.replace('test', 'parser_data')
        elif 'validation' in raw_data:
            save_path = raw_data.replace('validation', 'parser_data')

        pdb = save_path.split('/')[-1].split('.')[0]
        path_parts = save_path.split('/')[:-1]  # Split and remove last part
        base_path = os.path.join(*path_parts)   # Join the parts back together
        final_path = '/'+os.path.join(base_path, pdb[1:3], f"{pdb}.npz")
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        dump_structure_data(data, final_path)
    except Exception as e:
        print(f"Error processing {raw_data}: {str(e)}")

def main(data_path):
    raw_data_list = glob.glob(data_path + '*.cif.gz')

    # Determine the number of CPU cores to use
    num_cores = mp.cpu_count()
    
    # Create a pool of worker processes
    with mp.Pool(processes=num_cores) as pool:
        # Use tqdm to create a progress bar
        for _ in tqdm(pool.imap_unordered(process_file, raw_data_list), total=len(raw_data_list)):
            pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="train",
        help="Path to the data directory containing .cif.gz files"
    )
    args = parser.parse_args()
    data_path = args.data_path
    main(data_path)