
export PYTHONPATH=$PWD:$PYTHONPATH


python3 data/parser_dataset.py --data_path /ssd/dataset/pdb/test/
python3 data/parser_dataset.py --data_path /ssd/dataset/pdb/validation/
python3 data/parser_dataset.py --data_path /ssd/dataset/pdb/train/

