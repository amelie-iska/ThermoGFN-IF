
import os
import pickle
import secrets
import gzip
import pandas as pd
import numpy as np
from tqdm import tqdm
from Bio import pairwise2
from Bio import SeqIO
import math
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

restype_1to3 = {
    'A': 'ALA',
    'R': 'ARG',
    'N': 'ASN',
    'D': 'ASP',
    'C': 'CYS',
    'Q': 'GLN',
    'E': 'GLU',
    'G': 'GLY',
    'H': 'HIS',
    'I': 'ILE',
    'L': 'LEU',
    'K': 'LYS',
    'M': 'MET',
    'F': 'PHE',
    'P': 'PRO',
    'S': 'SER',
    'T': 'THR',
    'W': 'TRP',
    'Y': 'TYR',
    'V': 'VAL'}
AA_to_hydrophobicity_scores = {'A': 44, 'C': 50, 'D': -37, 'E': -12, 'F': 96, 'G': 0, 'H': -16, 'I': 100, 'K': -30, 'L': 99, 'M': 74, 'N': -35, 'P': -46, 'Q': -14, 'R': -20, 'S': -6, 'T': 13, 'V': 78, 'W': 90, 'Y': 57}
res_dict = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'ILE': 'I', 'LEU': 'L', 'PHE': 'F', 'PRO': 'P', 'MET': 'M', 'TRP': 'W', 'CYS': 'C', 'SER': 'S', 'THR': 'T', 'ASN': 'N', 'GLN': 'Q', 'TYR': 'Y', 'HIS': 'H', 'ASP': 'D', 'GLU': 'E', 'LYS': 'K', 'ARG': 'R'}



def cal_atomfea(seqence):
    all_for_assign = np.loadtxt("/home/tongpan/enzyme_predictor/all_assign.txt")
    xx = seqence
    x_p = np.zeros((len(xx), 7))
    x_h = np.zeros((len(xx), 1))
    for j in range(len(xx)):
        try:
            if restype_1to3[xx[j]] == 'ALA':
                x_p[j] = all_for_assign[0,:]
            elif restype_1to3[xx[j]] == 'CYS':
                x_p[j] = all_for_assign[1,:]
            elif restype_1to3[xx[j]] == 'ASP':
                x_p[j] = all_for_assign[2,:]
            elif restype_1to3[xx[j]] == 'GLU':
                x_p[j] = all_for_assign[3,:]
            elif restype_1to3[xx[j]] == 'PHE':
                x_p[j] = all_for_assign[4,:]
            elif restype_1to3[xx[j]] == 'GLY':
                x_p[j] = all_for_assign[5,:]
            elif restype_1to3[xx[j]] == 'HIS':
                x_p[j] = all_for_assign[6,:]
            elif restype_1to3[xx[j]] == 'ILE':
                x_p[j] = all_for_assign[7,:]
            elif restype_1to3[xx[j]] == 'LYS':
                x_p[j] = all_for_assign[8,:]
            elif restype_1to3[xx[j]] == 'LEU':
                x_p[j] = all_for_assign[9,:]
            elif restype_1to3[xx[j]] == 'MET':
                x_p[j] = all_for_assign[10,:]
            elif restype_1to3[xx[j]] == 'ASN':
                x_p[j] = all_for_assign[11,:]
            elif restype_1to3[xx[j]] == 'PRO':
                x_p[j] = all_for_assign[12,:]
            elif restype_1to3[xx[j]] == 'GLN':
                x_p[j] = all_for_assign[13,:]
            elif restype_1to3[xx[j]] == 'ARG':
                x_p[j] = all_for_assign[14,:]
            elif restype_1to3[xx[j]] == 'SER':
                x_p[j] = all_for_assign[15,:]
            elif restype_1to3[xx[j]] == 'THR':
                x_p[j] = all_for_assign[16,:]
            elif restype_1to3[xx[j]] == 'VAL':
                x_p[j] = all_for_assign[17,:]
            elif restype_1to3[xx[j]] == 'TRP':
                x_p[j] = all_for_assign[18,:]
            elif restype_1to3[xx[j]] == 'TYR':
                x_p[j] = all_for_assign[19,:]
        except:
            print("exception residue",xx[j])
    seqfea = np.concatenate((x_p,x_h),axis=1)













