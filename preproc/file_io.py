#!/usr/bin/env python
import pickle
import gzip
import os
import datetime
import time


def write_compressed_pickle(obj, filename, write_dir):
    '''Converts an object into byte representation and writes a compressed file.

    Args:
        obj: Generic Python object.
        filename: Name of file without file ending.
        write_dir (str): Output path.
    '''
    path = os.path.join(write_dir, f'{filename}.gz')
    pkl_obj = pickle.dumps(obj)
    try:
        with gzip.open(path, 'wb') as f:
            f.write(pkl_obj)
    except IOError as error:
        print(error)


def read_compressed_pickle(file_path):
    '''Reads a compressed binary file and return the object.

    Args:
        file_path (str): Path to the file (incl. filename)
    '''
    try:
        with gzip.open(file_path, 'rb') as f:
            pkl_obj = f.read()
            obj = pickle.loads(pkl_obj)
            return obj
    except IOError as error:
        print(error)


class Logger:
    '''Base class for handling logging information.
    '''
    def __init__(self, log_file_name, log_dir_path):

        # Set up file path
        self.log_file_name = log_file_name
        if os.path.isdir(log_dir_path):
            self.log_file_dir_path = log_dir_path
        else:
            raise IOError(f'Given logger directory invalid ({log_dir_path})')
        self.log_file_path = os.path.join(log_dir_path, log_file_name)

        self._create_new_file(self.log_file_path)

    def log(self):
        '''Implement to write an entry to the log file.
        '''
        raise NotImplementedError

    def _write_entry(self, entry_str):
        '''Writes a single entry string to the log file.
        '''
        try:
            with open(self.log_file_path, 'a') as file:
                file.write(entry_str)
        except IOError:
            print(f'Could not write to file ({self.log_file_path})')

    @staticmethod
    def _create_new_file(log_file_path):
        '''Creates a new file or overwrites an existing file.
        Adds creation time as comment.
        '''
        try: 
            with open(log_file_path, 'w') as file:
                file.write(f'# Creation time {datetime.datetime.now()}\n')
        except IOError:
            print(f'Could not write file ({log_file_path})')
        

class DSLALogger(Logger):
    '''Logger class for handling DSLA training process results logging.
    '''
    def __init__(self, log_file_name, log_dir_path):
        super().__init__(log_file_name, log_dir_path)

        # Add column discription
        descr_str = '# iter, sla_pos_acc, sla_neg_l1, da_avg, entry_pos_acc, '\
                    'entry_neg_l1, exit_pos_acc, exit_neg_l1\n'
        self._write_entry(descr_str)
    
    def log(self, iter, eval_dict):
        '''Writes the content of evaluation dictionary to the log file.
        Args:
            eval_dict: Evaluation dictionary with evaluation measures as keys
                       and sample means as value.
        '''

        sla_pos_acc = eval_dict['sla_pos_acc']
        sla_neg_l1 = eval_dict['sla_neg_l1']
        da_avg = eval_dict['da_avg']
        entry_pos_acc = eval_dict['entry_pos_acc']
        entry_neg_l1 = eval_dict['entry_neg_l1']
        exit_pos_acc = eval_dict['exit_pos_acc']
        exit_neg_l1 = eval_dict['exit_neg_l1']

        entry_str = f'{iter}, {sla_pos_acc:.6f}, {sla_neg_l1:.6f}, '\
                    f'{da_avg:.6f}, {entry_pos_acc:.6f}, {entry_neg_l1:.6f}, '\
                    f'{exit_pos_acc:.6f}, {exit_neg_l1:.6f}\n'

        self._write_entry(entry_str)
