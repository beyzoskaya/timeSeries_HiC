import pandas as pd
import numpy as np

def map_chromosome_data(gene_file, hic_file, chromosome_number, output_file):
    """
    Map HiC interactions with genes for a specific chromosome
    
    Parameters:
    gene_file: CSV file with gene data (including time series)
    hic_file: Space-separated HiC interaction file
    chromosome_number: Which chromosome to process
    output_file: Where to save the mapped data
    """
    genes_df = pd.read_csv(gene_file)
    
    genes_df['Chromosome'] = genes_df['Chromosome'].astype(str)
    chrom_genes = genes_df[genes_df['Chromosome'] == str(chromosome_number)].copy()
    
    time_columns = [col for col in genes_df.columns if col not in ['Gene', 'Chromosome', 'Start', 'End']]
    
    with open(hic_file, 'r') as f:
        hic_data = f.read().strip().split()
    hic_data = np.array(hic_data, dtype=float).reshape(-1, 3)
    
    def get_bin(position):
        return (position // 1000000) * 1000000
    
    chrom_genes['Bin'] = chrom_genes['Start'].apply(get_bin)
    
    mapped_data = []
    
    hic_dict = {(int(row[0]), int(row[1])): row[2] for row in hic_data}
    
    for i, gene1 in chrom_genes.iterrows():
        bin1 = gene1['Bin']
        gene1_name = gene1['Gene']
        
        for j, gene2 in chrom_genes.iterrows():
            bin2 = gene2['Bin']
            gene2_name = gene2['Gene']
            
            hic_freq = hic_dict.get((int(bin1), int(bin2)), 
                                  hic_dict.get((int(bin2), int(bin1)), 0))
            
            if hic_freq > 0:
                row_data = {
                    'Gene1': gene1_name,
                    'Gene1_Start': gene1['Start'],
                    'Gene1_End': gene1['End'],
                    'Gene1_Bin': bin1,
                    'Gene2': gene2_name,
                    'Gene2_Start': gene2['Start'],
                    'Gene2_End': gene2['End'],
                    'Gene2_Bin': bin2,
                    'HiC_Interaction': hic_freq
                }
                
                for time in time_columns:
                    row_data[f'Gene1_Time_{time}'] = gene1[time]
                    row_data[f'Gene2_Time_{time}'] = gene2[time]
                
                mapped_data.append(row_data)
    
    result_df = pd.DataFrame(mapped_data)
    result_df.to_csv(output_file, index=False)
    print(f"Mapped data saved to {output_file}")
    
    print(f"\nSummary for Chromosome {chromosome_number}:")
    print(f"Number of genes: {len(chrom_genes)}")
    print(f"Number of mapped interactions: {len(mapped_data)}")
    
    return result_df

if __name__ == "__main__":
    gene_file = "processed_mRNA_with_chromosomes.csv"  
    hic_file = "mESC_1mb_converted/HindIII_mESC.nor.chr2_1mb_contact_map.txt"  
    chromosome_number = "2"         
    output_file = "mapped/chromosome_2_mapped.csv"
    
    mapped_df = map_chromosome_data(gene_file, hic_file, chromosome_number, output_file)