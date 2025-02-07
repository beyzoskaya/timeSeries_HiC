import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

"""
One of the main uses of the GO is to perform enrichment analysis on gene sets. 
For example, given a set of genes that are up-regulated under certain conditions, 
an enrichment analysis will find which GO terms are over-represented (or under-represented) 
using annotations for that gene set.
"""

def get_available_databases():
    ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'
    response = requests.get(f'{ENRICHR_URL}/datasetStatistics')
    if not response.ok:
        raise Exception('Error getting databases')
    
    data = json.loads(response.text)
    print("\nAvailable databases containing 'GO' and 'Mouse':")
    for db in sorted(data.keys()):
        if 'GO' in db and 'Mouse' in db:
            print(db)
    return data.keys()

def get_enrichr_results(gene_list, database):
    ENRICHR_URL = 'https://maayanlab.cloud/Enrichr'
    
    genes_str = '\n'.join(gene_list)
    description = 'Mouse genes analysis'
    payload = {
        'list': (None, genes_str),
        'description': (None, description)
    }
    
    response = requests.post(f'{ENRICHR_URL}/addList', files=payload)
    if not response.ok:
        raise Exception('Error analyzing gene list')
    
    data = json.loads(response.text)
    user_list_id = data['userListId']
    
    response = requests.get(f'{ENRICHR_URL}/enrich',
                          params={'userListId': user_list_id,
                                 'backgroundType': database})
    if not response.ok:
        raise Exception('Error getting enrichment results')
    
    data = json.loads(response.text)
    return pd.DataFrame(data[database], columns=['Rank', 'Term', 'P-value', 'Z-score', 
                                               'Combined score', 'Overlapping genes',
                                               'Adjusted p-value', 'Old p-value', 
                                               'Old adjusted p-value'])

def plot_top_terms(results_dict, output_file='go_visualization_negative_corel.pdf'):
    plt.figure(figsize=(15, 10))
    
    num_categories = len(results_dict)
    current_plot = 1
    
    for name, df in results_dict.items():
        if not df.empty:
            plt.subplot(num_categories, 1, current_plot)
            
            top_terms = df.nsmallest(10, 'Adjusted p-value')
            
            bars = plt.barh(range(len(top_terms)), 
                          -np.log10(top_terms['Adjusted p-value']),
                          color=sns.color_palette("husl", n_colors=10))
            
            plt.yticks(range(len(top_terms)), 
                      [term[:50] + '...' if len(term) > 50 else term 
                       for term in top_terms['Term']])
            
            plt.xlabel('-log10(Adjusted p-value)')
            plt.title(f'Top {name} Terms')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width, bar.get_y() + bar.get_height()/2,
                        f'{width:.2f}',
                        ha='left', va='center', fontsize=8)
        
        current_plot += 1
    
    plt.tight_layout()
    plt.savefig('go_terms_barplot_negative_corel.pdf', bbox_inches='tight')
    plt.close()

def plot_volcano(results_dict, output_file='go_volcano_negative_corel.pdf'):
    plt.figure(figsize=(12, 8))
    
    colors = {'Biological Process': 'blue',
              'Molecular Function': 'green',
              'Cellular Component': 'red'}
    
    for name, df in results_dict.items():
        if not df.empty:
            enrichment_scores = df['Combined score'] / np.abs(df['Z-score'])
            
            plt.scatter(np.log2(enrichment_scores),
                       -np.log10(df['Adjusted p-value']),
                       alpha=0.6,
                       label=name,
                       color=colors[name])
            
            significant_terms = df[df['Adjusted p-value'] < 0.1]
            for idx, row in significant_terms.iterrows():
                plt.annotate(row['Term'][:30] + '...' if len(row['Term']) > 30 else row['Term'],
                           (np.log2(enrichment_scores[idx]), 
                            -np.log10(row['Adjusted p-value'])),
                           xytext=(5, 5),
                           textcoords='offset points',
                           fontsize=8)
    
    plt.axhline(y=-np.log10(0.1), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    
    plt.xlabel('log2(Enrichment Score)')
    plt.ylabel('-log10(Adjusted p-value)')
    plt.title('GO Terms Enrichment Analysis')
    plt.legend()
    
    plt.savefig('go_terms_volcano_negative_corel.pdf', bbox_inches='tight')
    plt.close()

def run_mouse_go_analysis():
    print("Checking available databases...")
    available_dbs = get_available_databases()
    
    # genes = [
    #  "Hist1h1b", "VIM", "P-63", "INMT", "ADAMTSL2", "Tnc", "FGF18", "Shisa3", "integrin subunit alpha 8", "Hist1h2ab", 
    #  "CD38", "MMP-3", "Lrp2", "ppia", "THTPA", "Vegf", "GATA-6", "ABCA3", "Kcnma1", "tfrc", "RAGE", "F13A1", "MCPt4",
    #  "FOXF2", "EPHA7", "AGER", "hmbs", "E2F8", "TGFB1", "TTF-1", "Claudin5", "GUCY1A2  sGC", "PRIM2", "tbp", "SFTP-D",
    #  "N-Cadherin", "Thy1", "Claudin 1", "Igfbp3", "EGFR", "ywhaz", "hprt", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS",
    #  "ABCG2", "AMACR"
    # ]
    genes = ['AMACR', 'ABCG2']
    

    databases = {
        'GO_Biological_Process_2021': 'Biological Process',
        'GO_Molecular_Function_2021': 'Molecular Function',
        'GO_Cellular_Component_2021': 'Cellular Component'
    }
    
    results = {}
    
    print("\nStarting mouse GO analysis...")
    for db, name in databases.items():
        print(f"\nAnalyzing {name} using {db}...")
        try:
            results[name] = get_enrichr_results(genes, db)
        except Exception as e:
            print(f"Warning: Error analyzing {name}: {str(e)}")
            results[name] = pd.DataFrame()
    
    with pd.ExcelWriter('mouse_go_analysis_results_negative_corel_genes.xlsx') as writer:
        for name, df in results.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=name, index=False)
    
    print("\nResults saved to 'mouse_go_analysis_results.xlsx'")
    
    for name, df in results.items():
        print(f"\nTop {name} terms:")
        if not df.empty:
            display_df = df[['Term', 'Adjusted p-value', 'Overlapping genes']]
            display_df = display_df[display_df['Adjusted p-value'] < 0.1]  # terms with p < 0.1
            if not display_df.empty:
                print(display_df.head())
            else:
                print("No significant terms found (p < 0.1)")
        else:
            print("No terms found")

    print("\nCreating visualizations...")
    plot_top_terms(results)
    plot_volcano(results)
    print("Visualizations saved as 'go_terms_barplot_negative_corel.pdf' and 'go_terms_volcano_negative_corel.pdf'")
    
    for name, df in results.items():
        print(f"\nTop {name} terms:")
        if not df.empty:
            display_df = df[['Term', 'Adjusted p-value', 'Overlapping genes']]
            display_df = display_df[display_df['Adjusted p-value'] < 0.1]
            if not display_df.empty:
                print(display_df.head())
            else:
                print("No significant terms found (p < 0.1)")
        else:
            print("No terms found")

if __name__ == "__main__":
    try:
        run_mouse_go_analysis()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        
        if isinstance(e, requests.exceptions.RequestException):
            print(f"API Error details: {e.response.text if hasattr(e, 'response') else 'No response text'}")
