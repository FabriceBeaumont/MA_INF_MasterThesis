from typing import List, Tuple, Any
import numpy as np
import pandas as pd

from os.path import exists, dirname
from os import makedirs
from pathlib import Path

class CSVWriter():

    def __init__(self, col_names: List[str], index_col_name: str = None, col_formats: List[str] = None, nr_rows: int = 1) -> None:
        self.col_names = np.array(col_names)
        
        # Construct the csv_table. This is a np.array which will be printed as csv-file.
        if col_formats is None:
            col_formats = ['f8'] * len(self.col_names)

        self.col_datatypes = np.dtype({"names": col_names, "formats": col_formats})
        self.csv_table = np.zeros((nr_rows), self.col_datatypes)
        
        # Define the index column.
        self.index_col = index_col_name
    
    def add_rows(self, nr_new_rows: int) -> None:
        new_rows = None
        if self.col_datatypes is None:
            new_rows = np.zeros((nr_new_rows))
        else:            
            new_rows = np.zeros((nr_new_rows), self.col_datatypes)

        self.csv_table = np.vstack((self.csv_table, new_rows))

    def add_entries(self, row_col_value_list: List[Tuple[int, str, Any]]) -> None:                
        max_row_ptr = max(row_col_value_list, key = lambda i : i[0])[0]
        nr_missing_rows = max_row_ptr - len(self.csv_table) + 1
        if nr_missing_rows > 0:
            self.add_rows(nr_missing_rows)

        for row_col_value_tuple in row_col_value_list:
            row_id, col_name, value = row_col_value_tuple
            self.csv_table[row_id][col_name] = value

    def get_column(self, col_name: str) -> np.array:
        return self.csv_table[col_name]

    def get_index_column(self) -> np.array:
        return self.csv_table[self.index_col]

    def add_entry(self, row_id: int, col_name: str, value: Any) -> None:                
        self.add_entries([(row_id, col_name, value)])

    def create_dir_if_not_existing(self, dir_name: str) -> None:
        if dir_name != '':
            if not exists(dir_name):
                makedirs(dir_name)

    def write_to_csv(self, file_path: str, header_row: str = '', write_mode: str = 'a', delimiter: str = '\t'):
        self.create_dir_if_not_existing(dirname(file_path))

        if not exists(file_path): 
            header_row = delimiter.join(self.col_names)
            write_mode = 'w'

        with open(file_path, write_mode) as f:
            np.savetxt(f, self.csv_table, fmt='%s', delimiter=delimiter, header=header_row)

    def convert_to_pandas_df(self):
        df = pd.DataFrame(self.csv_table, columns = self.col_names)
        return df

def write_to_csv_file(path: str, data_rows: np.array, write_mode: str = 'w', header_row: str = '', delimiter: str = '\t') -> None:
    with open(path, write_mode) as f:
        np.savetxt(f, data_rows, fmt='%s', delimiter=delimiter, header=header_row)

# requires pip<21.0.0,>=20.2.0, but you have pip 22.2.2 which is incompatible.
# class PathSettings(pydantic.BaseModel):    
#     path_names: np.array = np.array([]).astype(str) # np.array([str])
#     paths:      np.array = np.array([]).astype(Path) # np.array([Path])

#     def get_path(self, name: str) -> Path:
#         indices = np.where(self.path_names == name)[0]

#         if indices is not None:
#             return self.paths[indices[0]]

#     def add_path(self, path_name: str, path_str: str, inside_cwd: bool) -> None:
#         self.path_names = np.append(self.path_names, path_name)
        
#         path = Path(path_str)
#         if inside_cwd: 
#             path.parents = Path.cwd()

#         self.paths = np.append(self.paths, path)

def _example_usage_CSVWriter():
    col_dataset, col_wl_depth, col_avg_acc, col_std_dev, col_max_acc = 'Dataset', 'WL-Depth', 'Avg. Acc.', 'Std. Dev.', 'Max. Acc.'
    col_names =  np.array([col_dataset, col_wl_depth, col_avg_acc, col_std_dev, col_max_acc], dtype=np.str_)
    column_formats =      ['U50',     'i4',       'f4',       'f4',       'U1']
    csv_writer = CSVWriter(col_names, column_formats, 5)

    csv_writer.add_entries([(0, col_dataset, "TestDataset0")])
    csv_writer.add_entries([(1, col_wl_depth, 0), (2, col_avg_acc, 2.0), (3, col_std_dev, 3.0)])
    csv_writer.add_entries([(4, col_max_acc, 4.0)])
    csv_writer.add_entries([(4, col_dataset, "TestDataset4")])

    csv_writer.write_to_csv("test.csv")

def _example_usage_Path():   
    
    path_str_test_txt_file0     = "test_txt_file1.txt"
    path_str_test_nparr_file1   = "test_nparr_file1.npy"
    path_str_test_dir           = "TestDir"

    # Paths of txt files.
    path_test_txt_file0 = Path(path_str_test_txt_file0)
    if path_test_txt_file0.exists():
        print(f"Information on file {path_test_txt_file0}:")
        print(f"Content:    {path_test_txt_file0.read_text()}")
        print(f"Parent:     {path_test_txt_file0.parent}")
        print(f"Is file:    {path_test_txt_file0.is_file()}")
        print(f"File type:  {path_test_txt_file0.suffix}")

        path_test_txt_file0.unlink()
        print(f"Deleted file {path_test_txt_file0}.")
    else:
        path_test_txt_file0.touch()
        path_test_txt_file0.write_text("This file was created recently!")
        print(f"Created file {path_test_txt_file0}.")

    print()

    # Paths of directories. 
    path_test_dir = Path(path_str_test_dir)
    if path_test_dir.is_dir():
        path_test_dir.rmdir()
        print(f"Deleted directory {path_test_dir}.")                
    else:
        path_test_dir.mkdir()
        print(f"Deleted directory {path_test_dir}.")

    # Paths of npy files.
    path_str_test_nparr_file1 = path_test_dir / path_str_test_nparr_file1
    if path_str_test_nparr_file1.exists():
        print(f"Information on file {path_str_test_nparr_file1}:")
        print(f"Content:    {path_str_test_nparr_file1.read_text()}")
        print(f"Parent:     {path_str_test_nparr_file1.parent}")
        print(f"Is file:    {path_str_test_nparr_file1.is_file()}")
        print(f"File type:  {path_str_test_nparr_file1.suffix}")

        path_str_test_nparr_file1.unlink()
        print(f"Deleted file {path_str_test_nparr_file1}.")
    else:
        path_str_test_nparr_file1.touch()
        np.save(path_str_test_nparr_file1, np.array([1, 2, 3, 4], dtype=int))
        print(f"Created file {path_str_test_nparr_file1}.")



if __name__ == "__main__":
    # print("Hello world")
    # _example_usage_Path()
    pass
    

    
    
    