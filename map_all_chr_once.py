import pandas as pd
import numpy as np

def map_all_chromosomes(gene_file, hic_files_dict, output_file):
    """
    Map HiC interactions with genes across all chromosomes
    
    Parameters:
    gene_file: CSV file with gene data (including time series)
    hic_files_dict: Dictionary mapping chromosome numbers to their HiC files
    output_file: Where to save the mapped data
    """

    genes_df = pd.read_csv(gene_file)
    
    genes_df['Chromosome'] = genes_df['Chromosome'].astype(str)
    # time points for each gene with their chromsome name 
    time_columns = [col for col in genes_df.columns if col not in ['Gene', 'Chromosome', 'Start', 'End']]
    
    def get_bin(position):
        return (position // 1000000) * 1000000
    
    # find the bin
    genes_df['Bin'] = genes_df['Start'].apply(get_bin)
    
    all_hic_interactions = {}
    
    for chrom, hic_file in hic_files_dict.items():
        with open(hic_file, 'r') as f:
            hic_data = f.read().strip().split()
        hic_array = np.array(hic_data, dtype=float).reshape(-1, 3)
        
        for row in hic_array:
            bin1, bin2, interaction = int(row[0]), int(row[1]), row[2]
            all_hic_interactions[(chrom, bin1, chrom, bin2)] = interaction
            if bin1 != bin2: 
                all_hic_interactions[(chrom, bin2, chrom, bin1)] = interaction
    
    mapped_data = []
    
    for i, gene1 in genes_df.iterrows():
        for j, gene2 in genes_df.iterrows():
            chrom1 = gene1['Chromosome']
            chrom2 = gene2['Chromosome']
            bin1 = gene1['Bin']
            bin2 = gene2['Bin']
            
            interaction = all_hic_interactions.get((chrom1, bin1, chrom2, bin2), 0)
            
            # include if there's an interaction
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
    result_df.to_csv(output_file, index=False)
    
    print(f"\nMapping Summary:")
    print(f"Total number of genes: {len(genes_df)}")
    print(f"Number of mapped interactions: {len(mapped_data)}")
    print(f"Number of unique chromosomes: {len(genes_df['Chromosome'].unique())}")
    print(f"Chromosomes with interactions: {len(set(result_df['Gene1_Chromosome'].unique()))}")
    
    return result_df

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
        # Add all chromosome files
    }
    
    output_file = "mapped/complete_genome_mapping.csv"
    
    mapped_df = map_all_chromosomes(gene_file, hic_files_dict, output_file)