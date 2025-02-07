import pandas as pd
import requests
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform

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

def plot_gene_term_heatmap(results_dict, genes, min_pvalue=0.1, max_terms=20):
  
    term_size = 0.4  # inches per term
    gene_size = 1.0  # inches per gene
    
    all_terms = []
    gene_term_matrix = []
    pvalues = []  # Store p-values for sorting
    
    for category, df in results_dict.items():
        if not df.empty:
            significant_terms = df[df['Adjusted p-value'] < min_pvalue].sort_values('Adjusted p-value')
            significant_terms = significant_terms.head(max_terms)
            
            for _, row in significant_terms.iterrows():
                term = row['Term']
                # Handle overlapping genes whether it's a list or string
                if isinstance(row['Overlapping genes'], str):
                    overlapping_genes = row['Overlapping genes'].split(';')
                else:
                    overlapping_genes = row['Overlapping genes']
                
                overlapping_genes = [g.upper() for g in overlapping_genes]
                genes_upper = [g.upper() for g in genes]
                
                gene_matches = [1 if gene in overlapping_genes else 0 for gene in genes_upper]
                
                if sum(gene_matches) > 0:
                    term_name = f"{category[:2]}: {term}"
                    if len(term_name) > 50:
                        term_name = term_name[:47] + "..."
                    
                    all_terms.append(term_name)
                    gene_term_matrix.append(gene_matches)
                    pvalues.append(row['Adjusted p-value'])
    
    if gene_term_matrix and all_terms:
        sorted_indices = np.argsort(pvalues)
        gene_term_matrix = np.array(gene_term_matrix)[sorted_indices]
        all_terms = np.array(all_terms)[sorted_indices]
        
        fig_height = max(8, len(all_terms) * term_size)
        fig_width = max(10, len(genes) * gene_size)
        
        plt.figure(figsize=(fig_width, fig_height))
        
        sns.heatmap(gene_term_matrix,
                   xticklabels=genes,
                   yticklabels=all_terms,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Gene Present'},
                   center=0)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.title('Gene-Term Associations', pad=20, fontsize=12)
        plt.xlabel('Genes', fontsize=11)
        plt.ylabel('GO Terms', fontsize=11)
        
        plt.tight_layout()
        
        plt.savefig('gene_term_heatmap_relaxed_p_value.pdf', 
                   bbox_inches='tight', 
                   dpi=300)
        plt.close()
        
        print(f"Generated gene-term heatmap with {len(all_terms)} terms")
        
        with open('go_terms_description_relaxed_p_value.txt', 'w') as f:
            f.write("Full GO Terms Descriptions:\n\n")
            for term in all_terms:
                f.write(f"{term}\n")
        
        print("Saved full term descriptions to 'go_terms_description.txt'")
    else:
        print("No significant associations found for heatmap")

def plot_go_term_clusters(results_dict, genes, min_pvalue=0.1, max_terms=30):
    term_info = []
    gene_term_matrix = []

    for category, df in results_dict.items():
        if not df.empty:
            significant_terms = df[df['Adjusted p-value'] < min_pvalue].sort_values('Adjusted p-value')
            significant_terms = significant_terms.head(max_terms)
            
            for _, row in significant_terms.iterrows():
                if isinstance(row['Overlapping genes'], str):
                    overlapping_genes = row['Overlapping genes'].split(';')
                else:
                    overlapping_genes = row['Overlapping genes']
                
                overlapping_genes = [g.upper() for g in overlapping_genes]
                genes_upper = [g.upper() for g in genes]
                
                gene_matches = [1 if gene in overlapping_genes else 0 for gene in genes_upper]
                
                if sum(gene_matches) > 0:
                    term_info.append({
                        'term': row['Term'],
                        'category': category,
                        'p_value': row['Adjusted p-value']
                    })
                    gene_term_matrix.append(gene_matches)
    
    if gene_term_matrix:
        gene_term_matrix = np.array(gene_term_matrix)
        
        # Calculate linkage
        gene_linkage = linkage(pdist(gene_term_matrix.T), method='average')
        term_linkage = linkage(pdist(gene_term_matrix), method='average')
        
        plt.figure(figsize=(15, 10))
        
        ax_heatmap = plt.axes([0.3, 0.1, 0.6, 0.6])  # [left, bottom, width, height]
        
        gene_order = dendrogram(gene_linkage)['leaves']
        term_order = dendrogram(term_linkage)['leaves']
        
        matrix_ordered = gene_term_matrix[term_order][:, gene_order]
        
        sns.heatmap(matrix_ordered,
                   cmap='YlOrRd',
                   cbar_kws={'label': 'Association'},
                   ax=ax_heatmap)
        
        ordered_genes = [genes[i] for i in gene_order]
        ordered_terms = [f"{term_info[i]['category'][:2]}: {term_info[i]['term'][:30]}..." 
                        for i in term_order]
        
        ax_heatmap.set_xticks(np.arange(len(ordered_genes)) + 0.5)
        ax_heatmap.set_yticks(np.arange(len(ordered_terms)) + 0.5)
        
        ax_heatmap.set_xticklabels(ordered_genes, rotation=45, ha='right')
        ax_heatmap.set_yticklabels(ordered_terms, rotation=0)
        
        ax_dendro_top = plt.axes([0.3, 0.7, 0.48, 0.2])
        dendrogram(gene_linkage, ax=ax_dendro_top)
        ax_dendro_top.set_xticks([])
        ax_dendro_top.set_yticks([])
        
        ax_dendro_left = plt.axes([0.1, 0.1, 0.2, 0.6])
        dendrogram(term_linkage, orientation='left', ax=ax_dendro_left)
        ax_dendro_left.set_xticks([])
        ax_dendro_left.set_yticks([])
        
        plt.suptitle('GO Term and Gene Clusters', fontsize=14, y=0.95)
        
        plt.savefig('go_clusters.pdf', bbox_inches='tight', dpi=300)
        plt.close()
        
        with open('go_clusters_details.txt', 'w') as f:
            f.write("GO Term Clusters Analysis Details:\n\n")
            f.write("Genes in order of clustering:\n")
            f.write(", ".join(ordered_genes) + "\n\n")
            f.write("Terms in order of clustering:\n")
            for i in term_order:
                term = term_info[i]
                f.write(f"\nTerm: {term['term']}\n")
                f.write(f"Category: {term['category']}\n")
                f.write(f"P-value: {term['p_value']:.2e}\n")
        
        print("Generated cluster visualization")
        print("Saved details to 'go_clusters_details.txt'")
    else:
        print("No significant associations found for clustering")

def plot_enrichment_bubble(results_dict):
    plt.figure(figsize=(12, 8))
    
    colors = {'Biological Process': 'blue',
              'Molecular Function': 'green',
              'Cellular Component': 'red'}
    
    current_y = 0
    y_positions = {}
    
    for category, df in results_dict.items():
        if not df.empty:
            significant_terms = df[df['Adjusted p-value'] < 0.1].head(10)
            
            # Store y positions for each term
            term_positions = range(current_y, current_y + len(significant_terms))
            y_positions.update({term: pos for term, pos in 
                              zip(significant_terms['Term'], term_positions)})
            
            plt.scatter(x=-np.log10(significant_terms['Adjusted p-value']),
                       y=term_positions,
                       s=significant_terms['Combined score'],
                       alpha=0.6,
                       color=colors[category],
                       label=category)
            
            current_y += len(significant_terms) + 2
    
    plt.yticks(list(y_positions.values()),
               [f"{term[:50]}..." if len(term) > 50 else term 
                for term in y_positions.keys()])
    
    plt.xlabel('-log10(Adjusted p-value)')
    plt.title('GO Terms Enrichment Bubble Plot')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig('enrichment_bubble.pdf', bbox_inches='tight')
    plt.close()

def plot_term_network(results_dict, max_terms_per_category=8, min_pvalue=0.1):
    plt.figure(figsize=(15, 15))
    
    colors = {
        'Biological Process': '#ff7f0e',  # Orange
        'Molecular Function': '#2ca02c',  # Green
        'Cellular Component': '#1f77b4'   # Blue
    }
    
    significant_terms = []
    
    for category, df in results_dict.items():
        if not df.empty:
            # Sort by p-value and take top terms
            terms = df[df['Adjusted p-value'] < min_pvalue].sort_values('Adjusted p-value')
            terms = terms.head(max_terms_per_category)
            
            for _, row in terms.iterrows():
                significant_terms.append({
                    'term': row['Term'],
                    'category': category,
                    'p_value': row['Adjusted p-value'],
                    'score': row['Combined score']
                })
    
    if significant_terms:
        n_categories = len(colors)
        positions = {}
        
        for i, category in enumerate(colors.keys()):
            # Filter terms for this category
            category_terms = [term for term in significant_terms if term['category'] == category]
            n_terms = len(category_terms)
            
            if n_terms > 0:
                # Calculate radius for this category
                radius = 1.5 + i * 1.0  # Increasing radius for each category
                
                # Position terms in a circle
                for j, term in enumerate(category_terms):
                    angle = (2 * np.pi * j) / n_terms
                    x = radius * np.cos(angle)
                    y = radius * np.sin(angle)
                    positions[term['term']] = (x, y)
        
        for term in significant_terms:
            x, y = positions[term['term']]
            
            size = -np.log10(term['p_value']) * 500
            
            plt.scatter(x, y, 
                       s=size,
                       alpha=0.6,
                       c=colors[term['category']],
                       label=term['category'] if term['term'] == next(t['term'] for t in significant_terms if t['category'] == term['category']) else "")
            
            label_lines = []
            words = term['term'].split()
            current_line = words[0]
            
            for word in words[1:]:
                if len(current_line + ' ' + word) < 20:
                    current_line += ' ' + word
                else:
                    label_lines.append(current_line)
                    current_line = word
            label_lines.append(current_line)
            
            label_lines.append(f'p={term["p_value"]:.2e}')
            
            label = '\n'.join(label_lines)
            
            text_x = x * 1.0
            text_y = y * 1.0
    
            plt.annotate(label,
                        (x, y),
                        xytext=(text_x, text_y),
                        ha='center',
                        va='center',
                        fontsize=8,
                        bbox=dict(facecolor='white', 
                                alpha=0.7,
                                edgecolor='none',
                                pad=1))
        
        plt.legend(title="GO Categories", 
                  bbox_to_anchor=(1.05, 1), 
                  loc='upper left')
        
        plt.title('GO Terms Network\nNode size represents significance (-log10 p-value)',
                 pad=20)
        plt.axis('equal')
        plt.axis('off')
        
        plt.savefig('term_network_p_value_relaxed.pdf', 
                   bbox_inches='tight',
                   dpi=300)
        plt.close()
        
        print(f"Generated network plot with {len(significant_terms)} terms")
        
        with open('network_terms_details_p_value_relaxed.txt', 'w') as f:
            f.write("GO Terms Network Details:\n\n")
            for term in significant_terms:
                f.write(f"Term: {term['term']}\n")
                f.write(f"Category: {term['category']}\n")
                f.write(f"P-value: {term['p_value']:.2e}\n")
                f.write(f"Combined score: {term['score']:.2f}\n")
                f.write("\n")
        
        print("Saved term details to 'network_terms_details.txt'")
    else:
        print("No significant terms found for network plot")

def plot_significance_distribution(results_dict):
    plt.figure(figsize=(10, 6))
    
    data = []
    categories = []
    
    for category, df in results_dict.items():
        if not df.empty:
            data.extend(-np.log10(df['Adjusted p-value']))
            categories.extend([category] * len(df))
    
    if data:
        sns.violinplot(x=categories, y=data)
        
        plt.xticks(rotation=45)
        plt.xlabel('GO Category')
        plt.ylabel('-log10(Adjusted p-value)')
        plt.title('Distribution of Term Significance by Category')
        
        plt.tight_layout()
        plt.savefig('significance_distribution.pdf', bbox_inches='tight')
        plt.close()

def run_mouse_go_analysis():
    print("Checking available databases...")
    available_dbs = get_available_databases()
    
    genes = [
     "Hist1h1b", "VIM", "P-63", "INMT", "ADAMTSL2", "Tnc", "FGF18", "Shisa3", "integrin subunit alpha 8", "Hist1h2ab", 
     "CD38", "MMP-3", "Lrp2", "ppia", "THTPA", "Vegf", "GATA-6", "ABCA3", "Kcnma1", "tfrc", "RAGE", "F13A1", "MCPt4",
     "FOXF2", "EPHA7", "AGER", "hmbs", "E2F8", "TGFB1", "TTF-1", "Claudin5", "GUCY1A2  sGC", "PRIM2", "tbp", "SFTP-D",
     "N-Cadherin", "Thy1", "Claudin 1", "Igfbp3", "EGFR", "ywhaz", "hprt", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS",
     "ABCG2", "AMACR"
    ]

    #genes = ['AMACR', 'ABCG2']
    

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
    
    #with pd.ExcelWriter('mouse_go_analysis_results_negative_corel_genes.xlsx') as writer:
    #    for name, df in results.items():
    #        if not df.empty:
    #            df.to_excel(writer, sheet_name=name, index=False)
    
    #print("\nResults saved to 'mouse_go_analysis_results.xlsx'")
    
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
    #plot_top_terms(results)
    #plot_volcano(results)
    #plot_gene_term_heatmap(results, genes, min_pvalue=0.05, max_terms=15)
    #plot_gene_term_heatmap(results, genes, min_pvalue=0.1, max_terms=30)
    #print("plot_gene_term_heatmap done!'")
    #plot_enrichment_bubble(results)
    #print("plot_enrichment_bubble done!'")
    #plot_term_network(results, max_terms_per_category=5, min_pvalue=0.05)
    #plot_term_network(results, max_terms_per_category=8, min_pvalue=0.1)
    #print("plot_term_network done!'")
    #plot_significance_distribution(results)
    #print("Visualizations saved as 'go_terms_barplot_negative_corel.pdf' and 'go_terms_volcano_negative_corel.pdf'")
    plot_go_term_clusters(results, genes)
    print("New visualizations done!'")

    
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
