import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_gene_data(genes_df):
    required_columns = ['Gene', 'Chromosome', 'Start', 'End']
    
    missing_cols = [col for col in required_columns if col not in genes_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    time_columns = [col for col in genes_df.columns if col not in required_columns]
    if not time_columns:
        raise ValueError("No time point columns found")
    
    logging.info(f"Found {len(time_columns)} time points")
    logging.info(f"Total genes: {len(genes_df)}")
    
    return time_columns

def load_hic_data(hic_file):
    """Load and validate HiC interaction data"""
    try:
        with open(hic_file, 'r') as f:
            hic_data = f.read().strip().split()
        
        hic_array = np.array(hic_data, dtype=float).reshape(-1, 3)
        
        # Basic validations
        if hic_array.shape[1] != 3:
            raise ValueError(f"Expected 3 columns, found {hic_array.shape[1]}")
        
        if np.any(hic_array < 0):
            logging.warning(f"Negative values found in {hic_file}")
            
        return hic_array
        
    except Exception as e:
        logging.error(f"Error loading {hic_file}: {str(e)}")
        return None
    
def map_all_chromosomes(gene_file, hic_files_dict, output_file):
    """
    Enhanced mapping of HiC interactions with genes across chromosomes
    """
    logging.info("Starting chromosome mapping process...")
    
    genes_df = pd.read_csv(gene_file)
    time_columns = validate_gene_data(genes_df)
    
    genes_df['Chromosome'] = genes_df['Chromosome'].astype(str)
    
    bin_size = 1000000  # 1MB bins
    genes_df['Bin'] = genes_df['Start'].apply(lambda x: (x // bin_size) * bin_size)
    
    all_hic_interactions = {}
    processed_chroms = set()
    
    for chrom, hic_file in hic_files_dict.items():
        logging.info(f"Processing chromosome {chrom}")
        
        hic_array = load_hic_data(hic_file)
        if hic_array is None:
            continue
            
        for row in hic_array:
            bin1, bin2, interaction = int(row[0]), int(row[1]), row[2]
            all_hic_interactions[(chrom, bin1, chrom, bin2)] = interaction
            if bin1 != bin2:
                all_hic_interactions[(chrom, bin2, chrom, bin1)] = interaction
                
        processed_chroms.add(chrom)
        
    mapped_data = []
    skipped_interactions = 0
    
    for i, gene1 in genes_df.iterrows():
        for j, gene2 in genes_df.iterrows():
            if i >= j:  # Skip redundant pairs
                continue
                
            chrom1 = gene1['Chromosome']
            chrom2 = gene2['Chromosome']
            
            if chrom1 not in processed_chroms or chrom2 not in processed_chroms:
                skipped_interactions += 1
                continue
                
            bin1 = gene1['Bin']
            bin2 = gene2['Bin']
            
            interaction = all_hic_interactions.get((chrom1, bin1, chrom2, bin2), 0)
            
            if interaction > 0:
                row_data = {
                    'Gene1': gene1['Gene'],
                    'Gene1_Chromosome': chrom1,
                    'Gene1_Start': gene1['Start'],
                    'Gene1_End': gene1['End'],
                    'Gene1_Bin': bin1,
                    'Gene2': gene2['Gene'],
                    'Gene2_Chromosome': chrom2,
                    'Gene2_Start': gene2['Start'],
                    'Gene2_End': gene2['End'],
                    'Gene2_Bin': bin2,
                    'HiC_Interaction': interaction
                }
                
                for time in time_columns:
                    row_data[f'Gene1_Time_{time}'] = gene1[time]
                    row_data[f'Gene2_Time_{time}'] = gene2[time]
                
                mapped_data.append(row_data)

    result_df = pd.DataFrame(mapped_data)
    
    stats = {
        'Total_Genes': len(genes_df),
        'Mapped_Interactions': len(mapped_data),
        'Skipped_Interactions': skipped_interactions,
        'Processed_Chromosomes': len(processed_chroms),
        'Time_Points': len(time_columns)
    }
    
    output_path = Path(output_file)
    result_df.to_csv(output_file, index=False)
    
    stats_file = output_path.parent / (output_path.stem + '_stats.txt')
    with open(stats_file, 'w') as f:
        f.write("Mapping Statistics\n")
        f.write("=================\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
    
    logging.info(f"Mapping completed. Results saved to {output_file}")
    logging.info(f"Statistics saved to {stats_file}")
    
    return result_df, stats

if __name__ == "__main__":
    gene_file = "processed_mRNA_with_chromosomes.csv"
    
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
    
    output_file = "mapped/complete_genome_mapping_new.csv"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    mapped_df, mapping_stats = map_all_chromosomes(gene_file, hic_files_dict, output_file)

    print("\nMapping Summary:")
    for key, value in mapping_stats.items():
         print(f"{key}: {value}")