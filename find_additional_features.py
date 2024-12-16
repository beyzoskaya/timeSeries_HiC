import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.signal import find_peaks

def create_hic_matrix(hic_file, max_bin=None):
    """
    Create a matrix from HiC interaction data
    """
    hic_data = pd.read_csv(hic_file, sep='\s+', header=None, names=['bin1', 'bin2', 'interaction'])
    
    hic_data['bin1'] = (hic_data['bin1'] / 1000000).astype(int)
    hic_data['bin2'] = (hic_data['bin2'] / 1000000).astype(int)
    
    if max_bin is None:
        max_bin = max(hic_data['bin1'].max(), hic_data['bin2'].max()) + 1
    
    matrix = np.zeros((max_bin, max_bin))
    
    for _, row in hic_data.iterrows():
        i, j, value = int(row['bin1']), int(row['bin2']), row['interaction']
        matrix[i,j] = value
        matrix[j,i] = value  # symmetric matrix
    
    return matrix

def calculate_ab_compartments(hic_matrix):
    """
    Calculate A/B compartments using eigenvector decomposition
    """
    hic_matrix = np.array(hic_matrix, dtype=float)
  
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_matrix = np.corrcoef(hic_matrix)
    norm_matrix[np.isnan(norm_matrix)] = 0
    
    try:
        eigenvalues, eigenvectors = eigsh(norm_matrix, k=1, which='LM')
        compartment_eigenvector = eigenvectors[:, 0]
    except:
        compartment_eigenvector = np.zeros(len(hic_matrix))
    
    compartments = {}
    for i, value in enumerate(compartment_eigenvector):
        bin_pos = i * 1000000
        compartments[bin_pos] = 'A' if value > 0 else 'B'
    
    return compartments

def calculate_insulation_score(hic_matrix, window_size=5):
    """
    Calculate insulation score for TAD boundary detection
    """
    hic_matrix = np.array(hic_matrix, dtype=float)
    n = len(hic_matrix)
    insulation_scores = np.zeros(n)
    
    for i in range(window_size, n - window_size):
        window = hic_matrix[i-window_size:i+window_size, i-window_size:i+window_size]
        insulation_scores[i] = np.mean(window)
    
    insulation_scores = (insulation_scores - np.mean(insulation_scores)) / np.std(insulation_scores)
    return insulation_scores

def process_chromosome_features(input_file, hic_files_dict, output_file):
    
    df = pd.read_csv(input_file)
    
    df['Gene1_Compartment'] = 'Unknown'
    df['Gene2_Compartment'] = 'Unknown'
    df['Gene1_Insulation_Score'] = 0.0
    df['Gene2_Insulation_Score'] = 0.0
    df['Gene1_TAD_Boundary_Distance'] = 0.0
    df['Gene2_TAD_Boundary_Distance'] = 0.0
    
    for chrom in hic_files_dict.keys():
        print(f"Processing chromosome {chrom}...")
        
        hic_matrix = create_hic_matrix(hic_files_dict[chrom])
        
        if len(hic_matrix) == 0:
            print(f"Warning: Empty matrix for chromosome {chrom}")
            continue
            
        try:
            compartments = calculate_ab_compartments(hic_matrix)
            insulation_scores = calculate_insulation_score(hic_matrix)
            tad_boundaries = detect_tad_boundaries(insulation_scores)
            
            chrom_mask = (df['Gene1_Chromosome'] == str(chrom))
            for idx in df[chrom_mask].index:
                bin1 = df.loc[idx, 'Gene1_Bin']
                bin2 = df.loc[idx, 'Gene2_Bin']
                
                df.loc[idx, 'Gene1_Compartment'] = compartments.get(bin1, 'Unknown')
                df.loc[idx, 'Gene2_Compartment'] = compartments.get(bin2, 'Unknown')
                
                bin1_idx = int(bin1 // 1000000)
                bin2_idx = int(bin2 // 1000000)
                
                if bin1_idx < len(insulation_scores):
                    df.loc[idx, 'Gene1_Insulation_Score'] = insulation_scores[bin1_idx]
                if bin2_idx < len(insulation_scores):
                    df.loc[idx, 'Gene2_Insulation_Score'] = insulation_scores[bin2_idx]
         
                if len(tad_boundaries) > 0:
                    dist1 = min(abs(bin1_idx - b) for b in tad_boundaries)
                    dist2 = min(abs(bin2_idx - b) for b in tad_boundaries)
                    df.loc[idx, 'Gene1_TAD_Boundary_Distance'] = dist1
                    df.loc[idx, 'Gene2_TAD_Boundary_Distance'] = dist2
                    
        except Exception as e:
            print(f"Error processing chromosome {chrom}: {str(e)}")
            continue
    
    df.to_csv(output_file, index=False)
    print(f"Enhanced data saved to {output_file}")
    return df

def detect_tad_boundaries(insulation_scores, min_distance=5):
    """
    Detect TAD boundaries from insulation scores
    """
    boundaries, _ = find_peaks(-insulation_scores, distance=min_distance)
    return boundaries

if __name__ == "__main__":
    input_file = "mapped/complete_genome_mapping.csv"
    output_file = "mapped/enhanced_interactions.csv"
    
    hic_files_dict = {
        "1": "mESC_1mb_converted/HindIII_mESC.nor.chr1_1mb_contact_map.txt",
        "2": "mESC_1mb_converted/HindIII_mESC.nor.chr2_1mb_contact_map.txt",
        "3": "mESC_1mb_converted/HindIII_mESC.nor.chr3_1mb_contact_map.txt",
        "4": "mESC_1mb_converted/HindIII_mESC.nor.chr4_1mb_contact_map.txt",
        "5": "mESC_1mb_converted/HindIII_mESC.nor.chr5_1mb_contact_map.txt",
        "6": "mESC_1mb_converted/HindIII_mESC.nor.chr6_1mb_contact_map.txt",
        "7": "mESC_1mb_converted/HindIII_mESC.nor.chr7_1mb_contact_map.txt",
        "8": "mESC_1mb_converted/HindIII_mESC.nor.chr8_1mb_contact_map.txt",
        "9": "mESC_1mb_converted/HindIII_mESC.nor.chr9_1mb_contact_map.txt",
        "10": "mESC_1mb_converted/HindIII_mESC.nor.chr10_1mb_contact_map.txt",
        "11": "mESC_1mb_converted/HindIII_mESC.nor.chr11_1mb_contact_map.txt",
        "12": "mESC_1mb_converted/HindIII_mESC.nor.chr12_1mb_contact_map.txt",
        "13": "mESC_1mb_converted/HindIII_mESC.nor.chr13_1mb_contact_map.txt",
        "14": "mESC_1mb_converted/HindIII_mESC.nor.chr14_1mb_contact_map.txt",
        "15": "mESC_1mb_converted/HindIII_mESC.nor.chr15_1mb_contact_map.txt",
        "16": "mESC_1mb_converted/HindIII_mESC.nor.chr16_1mb_contact_map.txt",
        "17": "mESC_1mb_converted/HindIII_mESC.nor.chr17_1mb_contact_map.txt",
        "18": "mESC_1mb_converted/HindIII_mESC.nor.chr18_1mb_contact_map.txt",
        "19": "mESC_1mb_converted/HindIII_mESC.nor.chr19_1mb_contact_map.txt",
        "X": "mESC_1mb_converted/HindIII_mESC.nor.chrX_1mb_contact_map.txt",
    }
    
    
    enhanced_df = process_chromosome_features(input_file, hic_files_dict, output_file)
