import optuna
import numpy as np
import torch
from create_graph_and_embeddings_STGCN import TemporalGraphDataset
from STGCN_training import train_stgcn, evaluate_model_performance
import logging
import json
import traceback
from datetime import datetime

class Args:
    def __init__(self, trial=None):
        if trial is None:
            # Default parameters
            self.Kt = 3
            self.Ks = 3
            self.n_his = 3
            self.n_pred = 1
            self.blocks = [
                [32, 32, 32],
                [32, 48, 48],
                [48, 32, 1]
            ]
            self.act_func = 'glu'
            self.graph_conv_type = 'cheb_graph_conv'
            self.enable_bias = True
            self.droprate = 0.1
        else:
            # Parameters suggested by Optuna
            self.Kt = trial.suggest_int('Kt', 2, 3)
            self.Ks = 3  # Keep fixed as it's related to graph structure
            self.n_his = trial.suggest_int('n_his', 2, 4)
            self.n_pred = 1  # Keep fixed
            
            # keep fixed otherwise error on trials
            self.blocks = [
                [32, 32, 32],
                [32, 48, 48],
                [48, 32, 1]
            ]
            
            self.act_func = trial.suggest_categorical('act_func', ['glu', 'relu'])
            self.graph_conv_type = 'cheb_graph_conv'  # Keep fixed
            self.enable_bias = True  # Keep fixed
            self.droprate = trial.suggest_float('droprate', 0.1, 0.3)

class GridSearchOptimizer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.setup_logging()

    def setup_logging(self):
        # Set up file logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            filename=f'grid_search_log_{timestamp}.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
        
        # Also add console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

    def objective(self, trial):
        try:
            logging.info(f"\nStarting trial {trial.number}")
            
            # Graph construction parameters
            graph_params = {
                'hic_weight': trial.suggest_float('hic_weight', 0.2, 0.4),
                'compartment_weight': trial.suggest_float('compartment_weight', 0.05, 0.15),
                'tad_weight': trial.suggest_float('tad_weight', 0.03, 0.07),
                'ins_weight': trial.suggest_float('ins_weight', 0.03, 0.07),
                'expr_weight': trial.suggest_float('expr_weight', 0.4, 0.6),
                'cluster_sim_scale': trial.suggest_float('cluster_sim_scale', 0.8, 1.2)
            }
            logging.info(f"Graph parameters: {graph_params}")

            # Node2Vec parameters
            node2vec_params = {
                'dimensions': 32,
                'walk_length': trial.suggest_int('walk_length', 10, 30),
                'num_walks': trial.suggest_int('num_walks', 80, 120),
                'p': trial.suggest_float('p', 0.5, 2.0),
                'q': trial.suggest_float('q', 0.5, 2.0)
            }
            logging.info(f"Node2Vec parameters: {node2vec_params}")

            # Create Args instance
            args = Args(trial)
            args.Kt = trial.suggest_int('Kt', 2, 3)
            args.n_his = trial.suggest_int('n_his', 2, 4)
            args.droprate = trial.suggest_float('droprate', 0.1, 0.3)
            logging.info(f"Model parameters - Kt: {args.Kt}, n_his: {args.n_his}, droprate: {args.droprate}")

            logging.info("Creating dataset...")
            # Create dataset with current parameters
            dataset = TemporalGraphDataset(
                csv_file=self.csv_file,
                embedding_dim=node2vec_params['dimensions'],
                seq_len=args.n_his,
                pred_len=1
            )
            
            # Update args with number of vertices
            args.n_vertex = dataset.num_nodes
            logging.info(f"Dataset created with {args.n_vertex} vertices")

            logging.info("Starting model training...")
            # Train model
            model, val_sequences, val_labels, train_losses, val_losses = train_stgcn(
                dataset=dataset,
                val_ratio=0.2
            )
            logging.info("Model training completed")

            logging.info("Evaluating model performance...")

            metrics = evaluate_model_performance(model, val_sequences, val_labels, dataset)
        
            score = (
                metrics['Overall']['Pearson_Correlation'] * 0.4 +
                metrics['Gene']['Mean_Correlation'] * 0.4 +
                metrics['Temporal']['Mean_Direction_Accuracy'] * 0.2
            )

            logging.info(f"""
            Trial {trial.number} Results:
            Overall Correlation: {metrics['Overall']['Pearson_Correlation']:.4f}
            Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}
            Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}
            Final Score: {score:.4f}
            """)

            return score

        except Exception as e:
            logging.error(f"Error in trial {trial.number}:")
            logging.error(traceback.format_exc())
            raise e

    def run_optimization(self, n_trials=10):
        study = optuna.create_study(direction='maximize')
        
        try:
            study.optimize(self.objective, n_trials=n_trials)

            best_params = {
                'model_params': {
                    'Kt': study.best_params['Kt'],
                    'n_his': study.best_params['n_his'],
                    'droprate': study.best_params['droprate']
                },
                'graph_params': {
                    'hic_weight': study.best_params['hic_weight'],
                    'compartment_weight': study.best_params['compartment_weight'],
                    'tad_weight': study.best_params['tad_weight'],
                    'ins_weight': study.best_params['ins_weight'],
                    'expr_weight': study.best_params['expr_weight'],
                    'cluster_sim_scale': study.best_params['cluster_sim_scale']
                },
                'node2vec_params': {
                    'dimensions': study.best_params['dimensions'],
                    'walk_length': study.best_params['walk_length'],
                    'num_walks': study.best_params['num_walks'],
                    'p': study.best_params['p'],
                    'q': study.best_params['q']
                }
            }

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(f'best_params_{timestamp}.json', 'w') as f:
                json.dump(best_params, f, indent=4)

            logging.info(f"\nOptimization completed!")
            logging.info(f"Best trial score: {study.best_value:.4f}")
            logging.info(f"Best parameters: {json.dumps(best_params, indent=2)}")

            return best_params, study.best_value

        except Exception as e:
            logging.error("Error during optimization:")
            logging.error(traceback.format_exc())
            raise e


if __name__ == "__main__":
    print("Starting grid search optimization...")
    optimizer = GridSearchOptimizer(csv_file='mapped/enhanced_interactions.csv')
    
    try:
        best_params, best_score = optimizer.run_optimization(n_trials=10)
        print("\nOptimization completed successfully!")
        print(f"Best score: {best_score:.4f}")
        print("\nBest parameters:")
        print(json.dumps(best_params, indent=2))
    except Exception as e:
        print(f"Optimization failed: {str(e)}")

