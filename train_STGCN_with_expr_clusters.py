from create_graph_and_embeddings_STGCN import *
from STGCN_training import *
from evaluation import *
from utils import process_batch, calculate_correlation
from ontology_analysis import analyze_expression_levels

def filter_genes(dataset, gene_list):
    """Return a subset of the dataset containing only the specified genes."""
    filtered_node_map = {gene: idx for idx, gene in enumerate(gene_list)}

    filtered_dataset = dataset 
    filtered_dataset.node_map = filtered_node_map
    filtered_dataset.base_graph = {gene: dataset.base_graph[gene] for gene in gene_list}
    filtered_dataset.df = dataset.df[dataset.df['Gene1_clean'].isin(gene_list) | dataset.df['Gene2_clean'].isin(gene_list)]
    
    return filtered_dataset

def train_stgcn(dataset, val_ratio=0.2):

    args = Args()
    args.n_vertex = dataset.num_nodes

    #print(f"\nModel Configuration:")
    #print(f"Number of nodes: {args.n_vertex}")
    #print(f"Historical sequence length: {args.n_his}")
    #print(f"Block structure: {args.blocks}")

    # sequences with labels
    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nCreated {len(sequences)} sequences")

    # calculate GSO
    edge_index = sequences[0][0].edge_index
    edge_weight = sequences[0][0].edge_attr.squeeze() if sequences[0][0].edge_attr is not None else None

    adj = torch.zeros((args.n_vertex, args.n_vertex)) # symmetric matrix
    adj[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight # diagonal vs nondiagonal elements for adj matrix
    D = torch.diag(torch.sum(adj, dim=1) ** (-0.5))
    args.gso = torch.eye(args.n_vertex) - D @ adj @ D
    
    """
    # Split data into train and validation
    n_samples = len(sequences)
    n_train = int(n_samples * (1 - val_ratio))
    train_sequences = sequences[:n_train]
    train_labels = labels[:n_train]
    val_sequences = sequences[n_train:]
    val_labels = labels[n_train:]

    print(f"\nData Split:")
    print(f"Training sequences: {len(train_sequences)}")
    print(f"Validation sequences: {len(val_sequences)}")
    """
    # Random split
    n_samples = len(sequences)
    n_train = int(n_samples * (1 - val_ratio))
    indices = torch.randperm(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    train_sequences = [sequences[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_sequences = [sequences[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print("\n=== Training Data Statistics ===")
    print(f"Number of training sequences: {len(train_sequences)}")
    print(f"Number of validation sequences: {len(val_sequences)}")
    
    # Initialize STGCN
    model = STGCNGraphConvProjected(args, args.blocks, args.n_vertex)
    model = model.float() # convert model to float otherwise I am getting type error

    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    #criterion = nn.MSELoss()
    #criterion = temporal_loss_for_projected_model

    num_epochs = 100
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    save_dir = 'plottings_STGCN'
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    interaction_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_stats = []
        all_targets = []
        all_outputs = []

        for seq,label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            x,target = process_batch(seq, label)
            #x, _ = process_batch(seq, label)
            #print(f"Shape of x inside training: {x.shape}") # --> [1, 32, 5, 52]
            #print(f"Shape of target inside training: {target.shape}") # --> [1, 32, 1, 52]
            output = model(x)
            #print(f"Shape of output: {output.shape}") # --> [1, 32, 5, 52]
            #_, target = process_batch(seq, label)
            #target = target[:,:,-1:, :]
            #print(f"Shape of target: {target.shape}") # --> [32, 1, 52]

            # Don't take the last point for temporal loss !!!!!!
            #target = target[:, :, -1:, :]  # Keep only the last timestep
            #output = output[:, :, -1:, :]

            batch_stats.append({
                'target_range': [target.min().item(), target.max().item()],
                'output_range': [output.min().item(), output.max().item()],
                'target_mean': target.mean().item(),
                'output_mean': output.mean().item()
            })

            all_targets.append(target.detach().cpu().numpy())
            all_outputs.append(output.detach().cpu().numpy())
            #loss = temporal_pattern_loss(output[:, :, -1:, :], target, x)
            loss = temporal_loss_for_projected_model(
                output[:, :, -1:, :],
                target,
                x
            )
            if torch.isnan(loss):
                print("NaN loss detected!")
                print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # if epoch % 10 == 0:
        #         with torch.no_grad():
        #             mse = F.mse_loss(output, target)
        #             dir_loss = criterion(output, target, alpha=1, beta=0, gamma=0) - mse
        #             mag_loss = criterion(output, target, alpha=0, beta=1, gamma=0) - mse
        #             temp_loss = criterion(output, target, alpha=0, beta=0, gamma=1) - mse
        #             print(f"\nLoss Components:")
        #             print(f"MSE: {mse.item():.4f}")
        #             print(f"Direction: {dir_loss.item():.4f}")
        #             print(f"Magnitude: {mag_loss.item():.4f}")
        #             print(f"Temporal: {temp_loss.item():.4f}")

        model.eval()
        val_loss = 0
        val_loss_total = 0
        epoch_interaction_loss = 0
        with torch.no_grad():
            for seq,label in zip(val_sequences, val_labels):
                #x, _ = process_batch(seq, label)
                x,target = process_batch(seq, label)
                
                output = model(x)
                #_, target = process_batch(seq, label)

                # Don't take the last point for temporal loss!!!
                #target = target[:, :, -1:, :]  
                #output = output[:, :, -1:, :]

                #print(f"Shape of output in validation: {output.shape}") # --> [1, 32, 5, 52]
                #print(f"Shape of target in validation: {target.shape}") # --> [32, 1, 52]
                #target = target[:,:,-1:, :]
                #val_loss = temporal_pattern_loss(output[:, :, -1:, :], target, x)
                val_loss = temporal_loss_for_projected_model(output[:, :, -1:, :], target, x)
                val_loss_total += val_loss.item()

                output_corr = calculate_correlation(output)
                #print(f"Shape of output corr: {output_corr.shape}") # [32, 32]
                target_corr = calculate_correlation(target)
                #print(f"Shape of target corr: {target_corr.shape}") # [32, 32]
                int_loss = F.mse_loss(output_corr, target_corr)
                epoch_interaction_loss += int_loss.item()
        
        # Calculate average losses
        avg_train_loss = total_loss / len(train_sequences)
        avg_val_loss = val_loss_total / len(val_sequences)
        avg_interaction_loss = epoch_interaction_loss / len(val_sequences)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        interaction_losses.append(avg_interaction_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')
        #print(f'Interaction Loss: {avg_interaction_loss:.4f}\n')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'{save_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    checkpoint = torch.load(f'{save_dir}/best_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    #plt.savefig(f'{save_dir}/training_progress.png')
    plt.close()
    
    return model, val_sequences, val_labels, train_losses, val_losses  


def train_cluster_models(dataset, clusters, cluster_blocks):
    """Train separate models for each cluster."""
    cluster_models = {}
    cluster_metrics = {}

    for cluster_name, genes in clusters.items():
        print(f"\nTraining model for {cluster_name} with {len(genes)} genes.")
        
        # Create filtered dataset for the current cluster (reuse original dataset)
        cluster_dataset = filter_genes(dataset, genes)
        
        # Create Args with cluster-specific blocks
        args = Args(custom_blocks=cluster_blocks[cluster_name])
        
        # Train model for this cluster
        model, val_sequences, val_labels, train_losses, val_losses = train_stgcn(cluster_dataset, val_ratio=0.2)
        
        # Evaluate model
        metrics = evaluate_model_performance(model, val_sequences, val_labels, cluster_dataset)
        
        cluster_models[cluster_name] = model
        cluster_metrics[cluster_name] = metrics

        # Print the metrics as per the format you want
        print(f"Completed training for {cluster_name} cluster.")
        print(f"Metrics: {metrics['Overall']}")
        
        # Printing the performance summary as you requested:
        print("\nModel Performance Summary:")
        
        # Overall Metrics
        print("\nOverall Metrics:")
        for metric, value in metrics['Overall'].items():
            print(f"{metric}: {value:.4f}")

        # Gene Performance
        print("\nGene Performance:")
        print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
        print(f"Best Performing Genes: {', '.join(metrics['Gene']['Best_Genes'])}")

        # Temporal Performance
        print("\nTemporal Performance:")
        print(f"Time-lagged Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
        print(f"DTW Distance: {metrics['Temporal']['Mean_DTW_Distance']:.4f}")
        print(f"Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}")
        print(f"Change Magnitude Ratio: {metrics['Temporal']['Change_Magnitude_Ratio']:.4f}")

    return cluster_models, cluster_metrics


class Args:
    def __init__(self, custom_blocks=None):
        self.Kt = 3  # temporal kernel size
        self.Ks = 3  # spatial kernel size
        self.n_his = 3  # historical sequence length
        self.n_pred = 1
        self.blocks = custom_blocks if custom_blocks else [
            [32, 32, 32],    
            [32, 16, 16],    
            [16, 8, 1]
        ]
        self.act_func = 'glu'
        self.graph_conv_type = 'graph_conv'
        self.enable_bias = True
        self.droprate = 0

def main():
    # Initialize the dataset once
    dataset = TemporalGraphDataset(
        csv_file='mapped/enhanced_interactions.csv',
        embedding_dim=32,
        seq_len=3,
        pred_len=1
    )

    clusters, gene_expressions = analyze_expression_levels(dataset)
    
    cluster_blocks = {
        'high_expr': [
            [32, 48, 48],    # Wider for high expression genes
            [48, 32, 32],
            [32, 16, 1]
        ],
        'medium_expr': [
            [32, 32, 32],    # Standard for medium expression
            [32, 24, 24],
            [24, 16, 1]
        ],
        'low_expr': [
            [32, 24, 24],    
            [24, 16, 16],
            [16, 8, 1]
        ]
    }

    cluster_models, cluster_metrics = train_cluster_models(dataset, clusters, cluster_blocks)

    torch.save(cluster_models, 'plottings_STGCN/cluster_models.pth')
    print("\nCluster-based training completed. Models and metrics saved.")

if __name__ == "__main__":
    main()
