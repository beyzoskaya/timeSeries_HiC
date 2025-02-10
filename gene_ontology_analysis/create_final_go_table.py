import pandas as pd
import os
from distinct_temporal_patterns_go import clean_gene_name

def extract_genes_from_overlapping(overlap_str):
    return [gene.strip().strip("'") for gene in overlap_str.strip('[]').split(',')]

def create_master_table():
    files_info = {
        'general': {
            'path': '/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/mouse_go_analysis_results.xlsx',
            'sheets': ['Biological Process', 'Molecular Function', 'Cellular Component']
        },
        'negative_corr': {
            'path': '/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/mouse_go_analysis_results_negative_corel_genes.xlsx',
            'sheets': ['Biological Process', 'Molecular Function', 'Cellular Component']
        },
        'compartment': {
            'path': '/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/GO_results_compartments/compartment_go_analysis.xlsx',
            'sheets': ['Compartment_A_Biological', 'Compartment_A_Molecular', 
                      'Compartment_B_Biological', 'Compartment_B_Molecular']
        },
        'temporal': {
            'path': '/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/GO_results_temporal_patterns/temporal_go_analysis_mouse_genes_analysis_description.xlsx',
            'sheets': ['Cluster_0_Biological', 'Cluster_0_Molecular',  # Upregulated
                      'Cluster_1_Biological', 'Cluster_1_Molecular',  # Stable
                      'Cluster_2_Biological', 'Cluster_2_Molecular']  # Downregulated
        },
        'tad': {
            'path': '/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/GO_results_TAD/TAD_boundries_go_analyzes.xlsx',
            'sheets': ['Strong_Boundaries_GO_Biologi', 'Strong_Boundaries_GO_Molecul',
                      'Weak_Boundaries_GO_Biologi', 'Weak_Boundaries_GO_Molecul']
        },
        'tissue': {
            'path': '/Users/beyzakaya/Desktop/temporal gene/gene_ontology_analysis/GO_results_tissue/tissue_go_results.xlsx',
            'sheets': ['Mouse_Gene_Atlas']
        }
    }

    all_genes = set()
    for analysis_type, info in files_info.items():
        try:
            for sheet in info['sheets']:
                df = pd.read_excel(info['path'], sheet_name=sheet)
                for genes_str in df['Overlapping genes']:
                    genes = extract_genes_from_overlapping(genes_str)
                    all_genes.update(genes)
        except Exception as e:
            print(f"Error processing {analysis_type}: {e}")

    master_df = pd.DataFrame(index=sorted(all_genes))
    master_df.index.name = 'Gene'

    # Add columns for different analyses
    master_df['Temporal_Pattern'] = ''  # Up(0)/Stable(1)/Down(2)
    master_df['Compartment'] = ''       # A/B
    master_df['TAD_Boundary'] = ''      # Strong/Weak
    master_df['Correlation'] = ''       # Negative/Positive

    for analysis_type, info in files_info.items():
        for sheet in info['sheets']:
            try:
                df = pd.read_excel(info['path'], sheet_name=sheet)
                sig_terms = df[df['Adjusted p-value'] < 0.05]  # Only significant terms
            
                col_name = f"{analysis_type}_{sheet.replace(' ', '_')}"
                master_df[col_name] = ''
                
                for _, row in sig_terms.iterrows():
                    genes = extract_genes_from_overlapping(row['Overlapping genes'])
                    term_info = f"{row['Term']} (p={row['Adjusted p-value']:.2e})"
                    
                    for gene in genes:
                        if gene in master_df.index:
                            current = master_df.at[gene, col_name]
                            master_df.at[gene, col_name] = f"{current}; {term_info}" if current else term_info
                            
                            if 'Cluster_0' in sheet:
                                master_df.at[gene, 'Temporal_Pattern'] = 'Upregulated'
                            elif 'Cluster_1' in sheet:
                                master_df.at[gene, 'Temporal_Pattern'] = 'Stable'
                            elif 'Cluster_2' in sheet:
                                master_df.at[gene, 'Temporal_Pattern'] = 'Downregulated'
                            
                            if 'Compartment_A' in sheet:
                                master_df.at[gene, 'Compartment'] = 'A'
                            elif 'Compartment_B' in sheet:
                                master_df.at[gene, 'Compartment'] = 'B'
                            
                            if 'Strong_Boundaries' in sheet:
                                master_df.at[gene, 'TAD_Boundary'] = 'Strong'
                            elif 'Weak_Boundaries' in sheet:
                                master_df.at[gene, 'TAD_Boundary'] = 'Weak'

            except Exception as e:
                print(f"Error processing sheet {sheet} in {analysis_type}: {e}")

    return master_df

if __name__ == "__main__":
    master_table = create_master_table()
    master_table.to_excel("master_go_analysis.xlsx")
    print("Master table created successfully!")
    print(f"Total number of genes analyzed: {len(master_table)}")