import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mygene
import numpy as np
import networkx as nx
import matplotlib.patches as patches
import textwrap

def query_gene_info(genes):

    print("Querying gene information...")
    mg = mygene.MyGeneInfo()
    
    gene_info = mg.querymany(genes, 
                            scopes='symbol',
                            species='mouse',
                            fields=['go.BP', 'go.MF', 'go.CC', 'name'],
                            as_dataframe=True,
                            returnall=True)
    
    return gene_info['out']

def process_go_terms(gene_info_df):
    go_data = []
    
    for index, row in gene_info_df.iterrows():
        gene = row.name
        
        if 'go.BP' in row and isinstance(row['go.BP'], list):
            for term in row['go.BP']:
                if isinstance(term, dict):
                    go_data.append({
                        'Gene': gene,
                        'GO_ID': term.get('id', ''),
                        'GO_Term': term.get('term', ''),
                        'Category': 'Biological Process'
                    })
        
        if 'go.MF' in row and isinstance(row['go.MF'], list):
            for term in row['go.MF']:
                if isinstance(term, dict):
                    go_data.append({
                        'Gene': gene,
                        'GO_ID': term.get('id', ''),
                        'GO_Term': term.get('term', ''),
                        'Category': 'Molecular Function'
                    })
        
        if 'go.CC' in row and isinstance(row['go.CC'], list):
            for term in row['go.CC']:
                if isinstance(term, dict):
                    go_data.append({
                        'Gene': gene,
                        'GO_ID': term.get('id', ''),
                        'GO_Term': term.get('term', ''),
                        'Category': 'Cellular Component'
                    })
    
    return pd.DataFrame(go_data)

def get_comprehensive_categories(gene_list):
    mg = mygene.MyGeneInfo()
    
    print("Mapping gene symbols...")
    mapped_genes = mg.querymany(gene_list,
                               scopes='symbol',
                               species='mouse',
                               fields='symbol',
                               verbose=True)
    
    print("\nGetting functional information...")
    gene_info = mg.querymany(gene_list,
                            scopes='symbol',
                            species='mouse',
                            fields=['name', 'go', 'pathway.kegg', 'pathway.reactome', 
                                   'interpro', 'summary'],
                            as_dataframe=True)
    
    gene_categories = {}
    unmapped_genes = []
    
    for gene in gene_list:
        if gene in gene_info.index:
            info = gene_info.loc[gene]
            categories = set()  
            
            if 'go' in info and isinstance(info['go'], dict):
                if 'go.BP' in info['go']:
                    bp_terms = info['go']['go.BP']
                    if isinstance(bp_terms, list):
                        for term in bp_terms:
                            if isinstance(term, dict):
                                categories.add(('Biological Process', term.get('term', '')))
                
                if 'go.MF' in info['go']:
                    mf_terms = info['go']['go.MF']
                    if isinstance(mf_terms, list):
                        for term in mf_terms:
                            if isinstance(term, dict):
                                categories.add(('Molecular Function', term.get('term', '')))

                if 'go.CC' in info['go']:
                    cc_terms = info['go']['go.CC']
                    if isinstance(cc_terms, list):
                        for term in cc_terms:
                            if isinstance(term, dict):
                                categories.add(('Cellular Component', term.get('term', '')))
            
            if 'pathway.kegg' in info and isinstance(info['pathway.kegg'], list):
                for pathway in info['pathway.kegg']:
                    if isinstance(pathway, dict):
                        categories.add(('KEGG Pathway', pathway.get('name', '')))
            
            if 'pathway.reactome' in info and isinstance(info['pathway.reactome'], list):
                for pathway in info['pathway.reactome']:
                    if isinstance(pathway, dict):
                        categories.add(('Reactome Pathway', pathway.get('name', '')))
            
            gene_categories[gene] = list(categories)
        else:
            unmapped_genes.append(gene)
    
    if unmapped_genes:
        print("\nWarning: The following genes could not be mapped:")
        for gene in unmapped_genes:
            print(f"- {gene}")
        print("\nConsider checking these gene symbols or using alternative names.")
    
    return gene_categories, unmapped_genes

def suggest_alternative_names(unmapped_genes):
    mg = mygene.MyGeneInfo()
    suggestions = {}
    
    for gene in unmapped_genes:
        try:
            results = mg.query(gene, species='mouse', size=5)
            if 'hits' in results and results['hits']:
                suggestions[gene] = [
                    {
                        'symbol': hit.get('symbol', ''),
                        'name': hit.get('name', ''),
                        'score': hit.get('_score', 0)
                    }
                    for hit in results['hits']
                ]
        except:
            continue
    
    return suggestions

def update_gene_list_with_alternatives(gene_list, alternative_names):
    updated_gene_list = []
    
    for gene in gene_list:
        updated_gene_list.append(gene)
        if gene in alternative_names:
            updated_gene_list.extend(alternative_names[gene])
    
    return updated_gene_list

def categorize_genes_functionally(gene_list):
    
    mg = mygene.MyGeneInfo()
    
    categories = {
    'Signal Transduction': [
        'signal', 'PI3K-Akt', 'MAPK', 'calcium signaling', 
        'oxytocin signaling', 'receptor', 'kinase',
        'GUCY', 'guanylate cyclase', 'signal transducer'
    ],
    'Metabolism & Transport': [
        'metabolism', 'metabolic', 'transport', 'small molecules',
        'protein modification', 'post-translational', 'HPRT'
    ],
    'Transcription & Gene Expression': [
        'transcription', 'RNA polymerase', 'gene expression',
        'transcriptional regulation', 'TP53', 'GATA', 'TTF-1',
        'transcription factor', 'FOXF2', 'P63', 'nuclear factor', 
        'replication'
    ],
    'Cancer & Disease Related': [
        'cancer', 'gastric cancer', 'leukemia', 'papillomavirus',
        'cardiomyopathy', 'diabetic', 'tumor', 'metastasis'
    ],
    'Cellular Processes': [
        'cell cycle', 'apoptosis', 'differentiation',
        'hemostasis', 'cell adhesion', 'cell junction',
        'cell differentiation', 'hemoglobin', 'erythropoiesis', 
        'blood cell differentiation'
    ],
    'RNA Regulation': [
        'microRNA', 'RNA', 'splicing', 'processing',
        'RNA polymerase', 'transcription'
    ],
    'Protein Processing': [
        'protein modification', 'post-translational',
        'proteolysis', 'protein transport', 'MMP'
    ],
    'Cell Structure & Adhesion': [
        'integrin', 'adhesion', 'cadherin', 'claudin',
        'tight junction', 'cell junction', 'extracellular matrix',
        'cytoskeleton', 'membrane protein'
    ],
    'Chromatin & DNA Organization': [
        'histone', 'chromatin', 'nucleosome',
        'DNA binding', 'chromosome organization',
        'H1', 'H2A', 'H2B', 'chromatin remodeling'
    ],
    'Growth & Development': [
        'growth factor', 'development', 'morphogenesis',
        'differentiation', 'VEGF', 'FGF', 'tissue development'
    ],
    'Lung-Specific Functions': [
        'surfactant', 'SFTP', 'lung', 'pulmonary',
        'respiratory', 'alveolar'
    ]
    }

    gene_info = mg.querymany(gene_list,
                            scopes='symbol',
                            species='mouse',
                            fields=['name', 'go', 'pathway.kegg', 'pathway.reactome'],
                            as_dataframe=True)
    
    categorized_genes = {cat: [] for cat in categories.keys()}
    uncategorized = []
    
    for gene in gene_list:
        if gene in gene_info.index:
            info = gene_info.loc[gene]
            categorized = False
            
            search_text = ''
        
            if 'pathway.kegg' in info and isinstance(info['pathway.kegg'], list):
                for pathway in info['pathway.kegg']:
                    if isinstance(pathway, dict):
                        search_text += pathway.get('name', '') + ' '
            
            if 'pathway.reactome' in info and isinstance(info['pathway.reactome'], list):
                for pathway in info['pathway.reactome']:
                    if isinstance(pathway, dict):
                        search_text += pathway.get('name', '') + ' '
            
            search_text = search_text.lower()
            for cat, keywords in categories.items():
                if any(keyword.lower() in search_text for keyword in keywords):
                    categorized_genes[cat].append(gene)
                    categorized = True
            
            if not categorized:
                uncategorized.append(gene)
        else:
            uncategorized.append(gene)
    
    categorized_genes['Uncategorized'] = uncategorized
    uncategorized_genes = categorized_genes['Uncategorized']
    suggestions = suggest_alternative_names(uncategorized_genes)

    for gene, alternative_symbols in suggestions.items():
        print(f"Alternative names for {gene}:")
        for suggestion in alternative_symbols:
            print(f"- {suggestion['symbol']} ({suggestion['name']})")
    
    return categorized_genes

def visualize_pathway_based_categories(categorized_genes):
    category_sizes = {cat: len(genes) for cat, genes in categorized_genes.items()}
    
    plt.figure(figsize=(12, 6))
    
    categories = list(category_sizes.keys())
    sizes = list(category_sizes.values())
    
    sns.barplot(x=sizes, y=categories)
    plt.title('Gene Distribution Across Functional Categories')
    plt.xlabel('Number of Genes')
    
    plt.tight_layout()
    plt.show()
    
    print("\nDetailed Category Breakdown:")
    for category, genes in categorized_genes.items():
        print(f"\n{category} ({len(genes)} genes):")
        if genes:
            print(", ".join(genes))

def visualize_functional_categories(categorized_genes):
    # 1. Bar plot of category sizes
    plt.figure(figsize=(12, 6))
    category_sizes = {cat: len(genes) for cat, genes in categorized_genes.items()}
    
    sns.barplot(x=list(category_sizes.values()), 
                y=list(category_sizes.keys()),
                palette='viridis')
    plt.title('Number of Genes in Each Functional Category')
    plt.xlabel('Number of Genes')
    plt.tight_layout()
    plt.show()
    
    # 2. Network visualization of categories with more distance
    plt.figure(figsize=(15, 15))
    G = nx.Graph()
    
    # Add nodes for categories and genes
    for category, genes in categorized_genes.items():
        G.add_node(category, node_type='category')
        for gene in genes:
            G.add_node(gene, node_type='gene')
            G.add_edge(category, gene)
    
    # Adjust the spring layout for better spacing
    pos = nx.spring_layout(G, k=1.5, seed=42)  # Increased k for more spacing
    
    nx.draw_networkx_nodes(G, pos, 
                          nodelist=[n for n, d in G.nodes(data=True) if d['node_type']=='category'],
                          node_color='lightblue',
                          node_size=2500)
    
    nx.draw_networkx_nodes(G, pos,
                          nodelist=[n for n, d in G.nodes(data=True) if d['node_type']=='gene'],
                          node_color='lightgreen',
                          node_size=1200)
    
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8)
    
    nx.draw_networkx_labels(G, pos, font_size=10)
    
    plt.title('Gene Functional Category Network (Spaced Out)')
    plt.axis('off')
    plt.show()

def plot_gene_category_heatmap(categorized_genes):
    category_data = {gene: {cat: (gene in genes) for cat, genes in categorized_genes.items()} for gene in sum(categorized_genes.values(), [])}
    heatmap_df = pd.DataFrame(category_data).T
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df, annot=True, cmap="YlOrRd", linewidths=0.5)
    plt.title("Gene-Category Heatmap")
    plt.xlabel('Categories')
    plt.ylabel('Genes')
    plt.tight_layout()
    plt.show()

def analyze_functional_distribution(gene_categories):

    category_freq = {}
    for gene, categories in gene_categories.items():
        for cat_type, cat_name in categories:
            if (cat_type, cat_name) not in category_freq:
                category_freq[(cat_type, cat_name)] = 0
            category_freq[(cat_type, cat_name)] += 1
    
    df = pd.DataFrame([
        {'Category_Type': cat[0], 
         'Category_Name': cat[1], 
         'Frequency': freq} 
        for cat, freq in category_freq.items()
    ])
    
    df = df.sort_values('Frequency', ascending=False)
    
    category_types = df['Category_Type'].unique()
    
    plt.figure(figsize=(15, 10))
    for i, cat_type in enumerate(category_types, 1):
        plt.subplot(len(category_types), 1, i)
        data = df[df['Category_Type'] == cat_type].head(10)  
        sns.barplot(data=data, x='Frequency', y='Category_Name')
        plt.title(f'Top 10 {cat_type} Categories')
        plt.tight_layout()
    
    plt.show()
    
    return df

def plot_go_analysis(go_df):

    plt.figure(figsize=(15, 8))
    for i, category in enumerate(['Biological Process', 'Molecular Function', 'Cellular Component']):
        plt.subplot(1, 3, i+1)
        category_data = go_df[go_df['Category'] == category]['GO_Term'].value_counts().head(10)
        sns.barplot(x=category_data.values, y=category_data.index)
        plt.title(f'Top 10 {category} Terms')
        plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 8))
    pivot_table = pd.crosstab(go_df['Gene'], go_df['Category'])
    sns.heatmap(pivot_table, annot=True, cmap='YlOrRd')
    plt.title('Gene Annotation Distribution')
    plt.show()

def plot_clustered_gene_bubbles(categorized_genes):
    plt.figure(figsize=(12, 8))
    
    category_sizes = {cat: len(genes) for cat, genes in categorized_genes.items() if genes}
    
    x_positions = np.arange(len(category_sizes))
    sizes = np.array(list(category_sizes.values())) * 100  # Scale bubble size
    colors = sns.color_palette("husl", len(category_sizes))  # Unique colors

    for i, (category, size) in enumerate(category_sizes.items()):
        plt.scatter(x_positions[i], size, s=sizes[i], color=colors[i], alpha=0.6, label=category, edgecolors="black")

        gene_names = ", ".join(categorized_genes[category][:5])  
        plt.text(x_positions[i], size, gene_names, fontsize=9, ha='center', va='bottom', rotation=30)

    plt.xticks(x_positions, category_sizes.keys(), rotation=45, ha="right")
    plt.ylabel("Number of Genes")
    plt.title("Clustered Gene Categories - Bubble Plot")
    plt.legend(loc='upper right', fontsize=8, title="Categories")
    plt.tight_layout()
    plt.savefig('mygene_functional_categorization/functional_bubble_categories.png')
    plt.show()

def plot_separate_networks(categorized_genes):
    num_categories = len(categorized_genes)

    cols = min(3, num_categories)  # Max 3 columns per row
    rows = -(-num_categories // cols) 

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))  # Dynamic sizing
    axes = axes.flatten() if num_categories > 1 else [axes]  

    for i, (category, genes) in enumerate(categorized_genes.items()):
        ax = axes[i]
        
        G = nx.Graph()
        G.add_node(category, node_type='category')
        for gene in genes:
            G.add_node(gene, node_type='gene')
            G.add_edge(category, gene)

        pos = nx.spring_layout(G, seed=42, k=0.5)  

        nx.draw_networkx_nodes(G, pos, ax=ax, 
                               nodelist=[category], node_color='lightblue', node_size=1500)
        nx.draw_networkx_nodes(G, pos, ax=ax, 
                               nodelist=genes, node_color='lightgreen', node_size=800)
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.8, alpha=0.5)

        nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

        ax.set_title(category, fontsize=10, fontweight='bold', pad=10) 

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)

    plt.subplots_adjust(top=0.9, hspace=0.6, wspace=0.4)
    plt.savefig('mygene_functional_categorization/functional_categories_all_genes.png')
    plt.show()


if __name__ == "__main__":
    gene_list = [
     "Hist1h1b", "VIM", "P-63", "INMT", "ADAMTSL2", "Tnc", "FGF18", "Shisa3", "integrin subunit alpha 8", "H2ac4", 
     "CD38", "Mmp3", "Lrp2", "ppia", "THTPA", "Pgf", "Gata6", "ABCA3", "Kcnma1", "tfrc", "RAGE", "F13A1", "MCPt4",
     "FOXF2", "EPHA7", "AGER", "hmbs", "E2F8", "TGFB1", "Ttf1", "Claudin5", "Gucy1a2", "PRIM2", "tbp", "SFTP-D",
     "N-Cadherin", "Thy1", "Claudin 1", "Igfbp3", "EGFR", "ywhaz", "Hprt1", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS",
     "ABCG2", "AMACR"
    ]

    alternative_names = {
    'integrin subunit alpha 8': ['Lama4', 'Itga8', 'Col4a2', 'Itgav', 'Col3a1'],
    'AGER': ['Ager'],
    'N-Cadherin': ['Cdh2', 'Ctnna2', 'Arhgap32', 'Kifap3', 'Ctnnb1'],
    }

    alternative_names_overall = {
    'Hist1h1b': ['H1f5'],
    'P-63': ['Or5p63'],
    'Shisa3': ['Shisa3'],  
    'integrin subunit alpha 8': ['Lama4', 'Itga8', 'Col4a2', 'Itgav', 'Col3a1'],
    'Hist1h2ab': ['H2ac4'],
    'MMP-3': ['Mmp16', 'Mmp11', 'Mmp3', 'Mmp13'],
    'Vegf': ['Angvq2', 'Angvq1', 'Vegfb', 'Vegfd', 'Pgf'],
    'GATA-6': ['Gata6os', 'Gata6'],
    'RAGE': ['Mok', 'Ager'],
    'FOXF2': ['Foxf2'],
    'TTF-1': ['Ccdc59', 'Ttf1', 'Foxe1', 'Nkx2-1', 'Baz2a'],
    'GUCY1A2  sGC': ['Gucy1a2'],
    'N-Cadherin': ['Cdh2', 'Ctnna2', 'Arhgap32', 'Kifap3', 'Ctnnb1'],
    'Claudin 1': ['Cldn1', 'Cldnd1', 'Cldn34b3', 'Cldn34a', 'Cldn34c2'],
    'hprt': ['Hma', 'Hprt-ps1', 'Hprt1']
    }

    updated_gene_list = update_gene_list_with_alternatives(gene_list, alternative_names)

    categorized_genes = categorize_genes_functionally(updated_gene_list)
    #visualize_pathway_based_categories(categorized_genes)
    #visualize_functional_categories(categorized_genes)
    #plot_gene_category_heatmap(categorized_genes)

    #plot_clustered_gene_bubbles(categorized_genes)
    plot_separate_networks(categorized_genes)
    
    #print("Getting comprehensive functional categories...")
    #categories, unmapped_genes = get_comprehensive_categories(gene_list)
    
    #if unmapped_genes:
    #    print("\nSearching for alternative gene names...")
    #    suggestions = suggest_alternative_names(unmapped_genes)
        
    #    print("\nSuggested alternatives for unmapped genes:")
    #    for gene, alternatives in suggestions.items():
    #        print(f"\n{gene} alternatives:")
    #        for alt in alternatives:
    #            print(f"- {alt['symbol']}: {alt['name']} (score: {alt['score']:.2f})")
    
    #if categories:
    #    print("\nAnalyzing functional distribution...")
    #    category_df = analyze_functional_distribution(categories)

    # gene_info = query_gene_info(gene_list)
    # print("\nGene information retrieved successfully")

    # go_results = process_go_terms(gene_info)
    # print(f"\nProcessed {len(go_results)} GO terms")

    # plot_go_analysis(go_results)

    # print("\nSummary Statistics:")
    # print("\nNumber of annotations per category:")
    # print(go_results['Category'].value_counts())

    # print("\nTop 5 GO terms in each category:")
    # for category in ['Biological Process', 'Molecular Function', 'Cellular Component']:
    #     print(f"\n{category}:")
    #     print(go_results[go_results['Category'] == category]['GO_Term'].value_counts().head())

    #print("Categorizing genes by function...")
    #functional_categories = categorize_genes_functionally(gene_list)
   
    #print("\nFunctional Categorization Results:")
    #for category, genes in functional_categories.items():
    #    print(f"\n{category} ({len(genes)} genes):")
    #    print(', '.join(genes))

    #print("\nCreating visualizations...")
    #visualize_functional_categories(functional_categories)
    












