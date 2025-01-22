import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from scipy.signal import find_peaks
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def create_hic_matrix(hic_file, max_bin=None):
    """Create and validate HiC matrix"""
    logging.info(f"Creating HiC matrix from {hic_file}")
    
    try:
        hic_data = pd.read_csv(hic_file, sep='\s+', header=None, 
                              names=['bin1', 'bin2', 'interaction'])

        hic_data['bin1'] = (hic_data['bin1'] / 1000000).astype(int)
        hic_data['bin2'] = (hic_data['bin2'] / 1000000).astype(int)
        
        if max_bin is None:
            max_bin = max(hic_data['bin1'].max(), hic_data['bin2'].max()) + 1
            
        matrix = np.zeros((max_bin, max_bin))
        
        for _, row in hic_data.iterrows():
            i, j, value = int(row['bin1']), int(row['bin2']), row['interaction']
            if 0 <= i < max_bin and 0 <= j < max_bin:
                matrix[i,j] = value
                matrix[j,i] = value
                
        logging.info(f"Created matrix of size {matrix.shape}")
        return matrix
        
    except Exception as e:
        logging.error(f"Error creating HiC matrix: {str(e)}")
        return None

def calculate_ab_compartments(hic_matrix, chrom):
    logging.info(f"Calculating A/B compartments for chromosome {chrom}")
    
    try:
        mask = ~np.all(hic_matrix == 0, axis=0)
        filtered_matrix = hic_matrix[mask][:, mask]
        
        # correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            norm_matrix = np.corrcoef(filtered_matrix)
        norm_matrix = np.nan_to_num(norm_matrix, 0)
        
        # eigenvectors
        eigenvalues, eigenvectors = eigsh(norm_matrix, k=1, which='LM')
        compartment_eigenvector = eigenvectors[:, 0]
        
        # mapping back to original indices
        orig_indices = np.where(mask)[0]
        compartments = {}
        
        for idx, orig_idx in enumerate(orig_indices):
            bin_pos = orig_idx * 1000000
            compartments[bin_pos] = 'A' if compartment_eigenvector[idx] > 0 else 'B'
            
        logging.info(f"Found {list(compartments.values()).count('A')} A and {list(compartments.values()).count('B')} B compartments")
        return compartments
        
    except Exception as e:
        logging.error(f"Error calculating compartments: {str(e)}")
        return {}

def calculate_insulation_score(hic_matrix, window_size=5):
    logging.info(f"Calculating insulation scores with window size {window_size}")
    
    try:
        hic_matrix = np.array(hic_matrix, dtype=float)
        n = len(hic_matrix)
        
        if window_size >= n:
            raise ValueError(f"Window size ({window_size}) must be smaller than matrix size ({n})")
        
        insulation_scores = np.zeros(n)
        valid_counts = np.zeros(n)
        
        for i in range(window_size, n - window_size):
            window = hic_matrix[i-window_size:i+window_size, 
                              i-window_size:i+window_size]
            
            if np.any(window):
                insulation_scores[i] = np.nanmean(window)
                valid_counts[i] = 1
        
        valid_mask = valid_counts > 0
        if np.any(valid_mask):
            scores_mean = np.mean(insulation_scores[valid_mask])
            scores_std = np.std(insulation_scores[valid_mask])
            if scores_std > 0:
                insulation_scores[valid_mask] = ((insulation_scores[valid_mask] - scores_mean) 
                                               / scores_std)
        
        logging.info(f"Calculated insulation scores for {np.sum(valid_mask)} bins")
        return insulation_scores, valid_mask
        
    except Exception as e:
        logging.error(f"Error calculating insulation scores: {str(e)}")
        return np.zeros(len(hic_matrix)), np.zeros(len(hic_matrix), dtype=bool)

def detect_tad_boundaries(insulation_scores, valid_mask, min_distance=5, prominence=0.1):
    logging.info("Detecting TAD boundaries")
    
    try:
        working_scores = insulation_scores.copy()
        working_scores[~valid_mask] = 0
        
        #local minima (TAD boundaries)
        boundaries, properties = find_peaks(-working_scores,
                                         distance=min_distance,
                                         prominence=prominence)
        
        # Filter weak boundaries
        strong_boundaries = boundaries[properties['prominences'] > prominence]
        
        logging.info(f"Found {len(strong_boundaries)} TAD boundaries")
        return strong_boundaries, properties
        
    except Exception as e:
        logging.error(f"Error detecting TAD boundaries: {str(e)}")
        return np.array([]), {}


def validate_chromatin_features(hic_matrix, compartments, insulation_scores, 
                              boundaries, chrom):
    metrics = {}
    
    metrics['hic_matrix'] = {
        'size': hic_matrix.shape,
        'sparsity': np.sum(hic_matrix == 0) / hic_matrix.size,
        'max_value': np.max(hic_matrix),
        'mean_value': np.mean(hic_matrix[hic_matrix > 0])
    }
    
    if compartments:
        comp_values = list(compartments.values())
        metrics['compartments'] = {
            'total_bins': len(compartments),
            'A_fraction': comp_values.count('A') / len(comp_values),
            'B_fraction': comp_values.count('B') / len(comp_values)
        }
    
    valid_scores = insulation_scores[insulation_scores != 0]
    metrics['insulation'] = {
        'valid_bins': len(valid_scores),
        'mean_score': np.mean(valid_scores),
        'std_score': np.std(valid_scores),
        'range': [np.min(valid_scores), np.max(valid_scores)]
    }
    
    if len(boundaries) > 0:
        boundary_distances = np.diff(boundaries)
        metrics['tad_boundaries'] = {
            'count': len(boundaries),
            'mean_distance': np.mean(boundary_distances),
            'min_distance': np.min(boundary_distances),
            'max_distance': np.max(boundary_distances)
        }
    
    logging.info(f"\nValidation metrics for chromosome {chrom}:")
    for feature, values in metrics.items():
        logging.info(f"\n{feature.upper()} metrics:")
        for key, value in values.items():
            logging.info(f"  {key}: {value}")
    
    return metrics

def save_validation_report(metrics, output_file):
    with open(output_file, 'w') as f:
        f.write("Chromatin Feature Validation Report\n")
        f.write("================================\n\n")
        
        for chrom, chrom_metrics in metrics.items():
            f.write(f"\nChromosome {chrom}\n")
            f.write("-" * 20 + "\n")
            
            for feature, values in chrom_metrics.items():
                f.write(f"\n{feature.upper()}:\n")
                for key, value in values.items():
                    f.write(f"  {key}: {value}\n")

def process_chromosome_features(input_file, hic_files_dict, output_file):
    df = pd.read_csv(input_file)
    
    df['Gene1_Compartment'] = 'Unknown'
    df['Gene2_Compartment'] = 'Unknown'
    df['Gene1_Insulation_Score'] = 0.0
    df['Gene2_Insulation_Score'] = 0.0
    df['Gene1_TAD_Boundary_Distance'] = 0.0
    df['Gene2_TAD_Boundary_Distance'] = 0.0

    validation_metrics = {}
    
    for chrom in hic_files_dict.keys():
        logging.info(f"Processing chromosome {chrom}...")

        hic_matrix = create_hic_matrix(hic_files_dict[chrom])
        
        if hic_matrix is None or hic_matrix.shape[0] == 0 or hic_matrix.shape[0] != hic_matrix.shape[1]:
            logging.warning(f"Invalid matrix shape for chromosome {chrom}")
            continue
            
        try:
            compartments = calculate_ab_compartments(hic_matrix, chrom)
            insulation_scores, valid_mask = calculate_insulation_score(hic_matrix)
            tad_boundaries, boundary_props = detect_tad_boundaries(insulation_scores, valid_mask)
            
            validation_metrics[chrom] = validate_chromatin_features(
                hic_matrix, compartments, insulation_scores, tad_boundaries, chrom
            )
            
            chrom_mask = (df['Gene1_Chromosome'] == str(chrom))
            for idx in df[chrom_mask].index:
                bin1 = df.loc[idx, 'Gene1_Bin']
                bin2 = df.loc[idx, 'Gene2_Bin']
            
                df.loc[idx, 'Gene1_Compartment'] = compartments.get(bin1, 'Unknown')
                df.loc[idx, 'Gene2_Compartment'] = compartments.get(bin2, 'Unknown')
                
                bin1_idx = int(bin1 // 1000000)
                bin2_idx = int(bin2 // 1000000)
                
                if bin1_idx < len(insulation_scores) and valid_mask[bin1_idx]:
                    df.loc[idx, 'Gene1_Insulation_Score'] = insulation_scores[bin1_idx]
                if bin2_idx < len(insulation_scores) and valid_mask[bin2_idx]:
                    df.loc[idx, 'Gene2_Insulation_Score'] = insulation_scores[bin2_idx]
                
                if len(tad_boundaries) > 0:
                    dist1 = min(abs(bin1_idx - b) for b in tad_boundaries)
                    dist2 = min(abs(bin2_idx - b) for b in tad_boundaries)
                    df.loc[idx, 'Gene1_TAD_Boundary_Distance'] = dist1
                    df.loc[idx, 'Gene2_TAD_Boundary_Distance'] = dist2
    
            logging.info(f"""
            Chromosome {chrom} processing complete:
            - Compartments assigned: {df[chrom_mask]['Gene1_Compartment'].value_counts().to_dict()}
            - Valid insulation scores: {sum(valid_mask)}
            - TAD boundaries found: {len(tad_boundaries)}
            """)
                    
        except Exception as e:
            logging.error(f"Error processing chromosome {chrom}: {str(e)}")
            continue
    
    df.to_csv(output_file, index=False)
    
    validation_file = output_file.replace('.csv', '_validation.txt')
    save_validation_report(validation_metrics, validation_file)

    feature_stats = {
        'total_genes': len(df),
        'compartment_distribution': df['Gene1_Compartment'].value_counts().to_dict(),
        'mean_insulation_score': df['Gene1_Insulation_Score'].mean(),
        'mean_tad_distance': df['Gene1_TAD_Boundary_Distance'].mean(),
        'unknown_compartments': (df['Gene1_Compartment'] == 'Unknown').sum()
    }
    
    logging.info("\nFeature Processing Summary:")
    for key, value in feature_stats.items():
        logging.info(f"{key}: {value}")
    
    return df, validation_metrics, feature_stats


def debug_gene_filtering(mapping_file, enhanced_file):
    logging.info("Starting gene filtering debug...")
    
    mapping_df = pd.read_csv(mapping_file)
    enhanced_df = pd.read_csv(enhanced_file)

    original_genes = set(mapping_df['Gene1'].unique())
    new_genes = set(enhanced_df['Gene1'].unique())
    missing_genes = original_genes - new_genes
    
    logging.info(f"\nGene Count Summary:")
    logging.info(f"Original genes: {len(original_genes)}")
    logging.info(f"New genes: {len(new_genes)}")
    logging.info(f"Missing genes: {len(missing_genes)}")
    
    if missing_genes:
        logging.info("\nMissing Gene Details:")
        for gene in missing_genes:
            original_info = mapping_df[mapping_df['Gene1'] == gene].iloc[0]
            logging.info(f"\nGene: {gene}")
            logging.info(f"Chromosome: {original_info['Gene1_Chromosome']}")
            logging.info(f"Bin: {original_info['Gene1_Bin']}")
            logging.info(f"Original HiC Interaction: {original_info['HiC_Interaction']}")


if __name__ == "__main__":
    input_file = "mapped/complete_genome_mapping_new.csv"
    output_file = "mapped/enhanced_interactions_new_new.csv"
    
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
    
    
    df, validation_metrics, feature_stats = process_chromosome_features(
        input_file, 
        hic_files_dict, 
        output_file
    )

    debug_gene_filtering(input_file, output_file)

    
