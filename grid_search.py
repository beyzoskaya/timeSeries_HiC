import optuna
import numpy as np
import torch
from create_graph_and_embeddings_STGCN import TemporalGraphDataset
from STGCN_training import train_stgcn, evaluate_model_performance
import logging
import json
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
            
            # Suggested block structure
            channels = trial.suggest_int('channels', 32, 64)
            mid_channels = trial.suggest_int('mid_channels', 40, 72)
            self.blocks = [
                [channels, channels, channels],
                [channels, mid_channels, mid_channels],
                [mid_channels, channels, 1]
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.basicConfig(
            filename=f'grid_search_log_{timestamp}.txt',
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )

    def objective(self, trial):
        graph_params = {
            'hic_weight': trial.suggest_float('hic_weight', 0.2, 0.4),
            'compartment_weight': trial.suggest_float('compartment_weight', 0.05, 0.15),
            'tad_weight': trial.suggest_float('tad_weight', 0.03, 0.07),
            'ins_weight': trial.suggest_float('ins_weight', 0.03, 0.07),
            'expr_weight': trial.suggest_float('expr_weight', 0.4, 0.6),
            'cluster_sim_scale': trial.suggest_float('cluster_sim_scale', 0.8, 1.2)
        }

        node2vec_params = {
            'dimensions': trial.suggest_int('dimensions', 24, 64),
            'walk_length': trial.suggest_int('walk_length', 10, 30),
            'num_walks': trial.suggest_int('num_walks', 80, 120),
            'p': trial.suggest_float('p', 0.5, 2.0),
            'q': trial.suggest_float('q', 0.5, 2.0)
        }

        model_params = {
            'Kt': trial.suggest_int('Kt', 2, 3),
            'droprate': trial.suggest_float('droprate', 0.1, 0.3),
            'n_his': trial.suggest_int('n_his', 2, 4),
            'seq_len': trial.suggest_int('seq_len', 2, 4)
        }

        try:
            dataset = TemporalGraphDataset(
                csv_file=self.csv_file,
                embedding_dim=node2vec_params['dimensions'],
                seq_len=model_params['seq_len'],
                pred_len=1,
                graph_params=graph_params,
                node2vec_params=node2vec_params
            )

            args = Args() #FIXED
            args.Kt = model_params['Kt']
            args.droprate = model_params['droprate']
            args.n_his = model_params['n_his']

            model, val_sequences, val_labels, _, _ = train_stgcn(
                dataset, 
                val_ratio=0.2,
                early_stopping_patience=5,  # Reduced for faster trials
                num_epochs=50  # Reduced for faster trials
            )

            # Evaluate model
            metrics = evaluate_model_performance(model, val_sequences, val_labels, dataset)

            score = (
                metrics['Overall']['Pearson_Correlation'] * 0.3 +
                metrics['Gene']['Mean_Correlation'] * 0.4 +
                metrics['Temporal']['Mean_Direction_Accuracy'] * 0.3
            )

            # Log results
            log_msg = f"""
            Trial {trial.number}:
            Graph params: {graph_params}
            Node2Vec params: {node2vec_params}
            Model params: {model_params}
            Metrics:
                Overall Correlation: {metrics['Overall']['Pearson_Correlation']:.4f}
                Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}
                Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}
                Composite Score: {score:.4f}
            """
            logging.info(log_msg)

            return score
        
        except Exception as e:
            logging.error(f"Error in trial {trial.number}: {str(e)}")
            return float('-inf')

    def run_optimization(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)

        # Save best parameters
        best_params = study.best_params
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'best_params_{timestamp}.json', 'w') as f:
            json.dump(best_params, f, indent=4)

        # Log best results
        logging.info(f"\nBest trial:")
        logging.info(f"Value: {study.best_value:.4f}")
        logging.info(f"Params: {study.best_params}")

        return study.best_params, study.best_value

if __name__ == "__main__":
    optimizer = GridSearchOptimizer(csv_file='mapped/enhanced_interactions.csv')
    best_params, best_score = optimizer.run_optimization(n_trials=100)
    print("Optimization completed!")
    print(f"Best score: {best_score:.4f}")
    print("Best parameters:", json.dumps(best_params, indent=2))


