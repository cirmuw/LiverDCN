import torch
import torch.nn as nn
import numpy as np

from torch.utils.data.dataset import TensorDataset
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
import mlflow
import mlflow.pytorch

from datasets import LiverPatchDS
from models import autoencoder


class DCN():
    def __init__(self, num_bottleneck=20, kmeans_cluster=10, pretrain_epochs=15, pretrain_model=None,
                 num_epochs=50, reg=0.5, mlflow_logging=True, mlflow_experiment='1', outputpath=None, outputsuffix='1',
                 patchespath='/root/scratch/small_clusteringpatches_2/', cuda='cuda', lr=0.001, inchannels=6,
                 debug_latents=False):
        self.kmeans_cluster = kmeans_cluster
        self.outputpath = outputpath
        self.outputsuffix = outputsuffix
        self.cuda = cuda
        self.lr = lr
        self.pretrain_model = pretrain_model
        self.mlflow_logging = mlflow_logging
        self.num_bottleneck = num_bottleneck
        self.criterion = nn.MSELoss()
        self.num_epochs = num_epochs
        self.reg = reg
        self.inchannels = inchannels
        self.pretrain_epochs = pretrain_epochs
        self.debug_latents = debug_latents

        self.ds = LiverPatchDS(patchespath)
        self.dataloader = torch.utils.data.DataLoader(self.ds, batch_size=64, shuffle=False, num_workers=16)

        if mlflow_logging:
            mlflow.end_run()
            mlflow.start_run(experiment_id=mlflow_experiment)
            mlflow.log_param('outprefix', self.outputsuffix)
            mlflow.log_param('outpath', outputpath)
            mlflow.log_param('patches', patchespath)
            mlflow.log_param('num_bottleneck', num_bottleneck)
            mlflow.log_param('lr', lr)
            mlflow.log_param('pretrain_epochs', pretrain_epochs)
            mlflow.log_param('kmeans_clusters', kmeans_cluster)
            mlflow.log_param('num_epochs', num_epochs)
            mlflow.log_param('reg', reg)

    def run_training(self):
        self.device = torch.device(self.cuda if torch.cuda.is_available() else 'cpu')

        self.model = autoencoder(inchannels=self.inchannels, num_bottleneck=self.num_bottleneck).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if self.pretrain_model is None:
            self.run_pretraining()
            torch.save(self.model.state_dict(), self.outputpath + '/pretrain_DCN_run_' + self.outputsuffix + '.pt')
            if self.mlflow_logging:
                mlflow.log_artifact(self.outputpath + '/pretrain_DCN_run_' + self.outputsuffix + '.pt')
            else:
                self.model.load_state_dict(torch.load(self.pretrain_model))
                self.model.eval()

        latents = self.get_all_latents(self.dataloader)
        kmeans = KMeans(n_clusters=self.kmeans_cluster, n_jobs=-1, random_state=0)
        kmeans.fit(latents)

        centers = self.run_finetuning(kmeans.labels_, kmeans.cluster_centers_)

        torch.save(self.model.state_dict(), self.outputpath + '/DCN_run_' + self.outputsuffix + '.pt')
        centers.dump(self.outputpath + '/DCN_KMeans_Centers_' + self.outputsuffix + '.pkl')

        if self.mlflow_logging:
            mlflow.log_artifact(self.outputpath + '/DCN_run_' + self.outputsuffix + '.pt')
            mlflow.log_artifact(self.outputpath + '/DCN_KMeans_Centers_' + self.outputsuffix + '.pkl')

    def run_pretraining(self):
        for epoch in range(self.pretrain_epochs):
            running_loss = 0.0
            for data in self.dataloader:
                inp = data[0].to(self.device)
                recon = self.model(inp)
                loss = self.criterion(recon, inp)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            if self.mlflow_logging:
                mlflow.log_metric(key='pretrain_loss', value=running_loss / len(self.dataloader), step=epoch)

        self.model.eval()

    def run_finetuning(self, labels, centers):
        # finetuning
        self.ds.setlabels(labels)

        for epoch in range(self.num_epochs):
            running_loss = 0
            self.model.train()

            # updating network
            for data in self.dataloader:
                inp = data[0].to(self.device)

                recon = self.model(inp)
                latent = self.model.get_latent(inp)
                center = centers[data[1]]
                center = torch.tensor(center).to(self.device)
                loss = self.kmeans_friendly_loss_function(recon, inp, latent, center, self.criterion, self.reg)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()

            self.model.eval()
            # updating cluster assignments
            new_labels = list()
            latents = self.get_all_latents(self.dataloader)

            if self.debug_latents:
                latents.dump('debug_latents/latents_' + str(epoch) + '.pkl')

            for l in latents:
                dists = [np.sqrt(np.sum((l - c) ** 2)) for c in centers]
                new_labels.append(np.argmin(dists))

            if self.mlflow_logging:
                mlflow.log_metric(key='finetune_loss', value=running_loss / len(self.dataloader), step=epoch)


            self.ds.setlabels(new_labels)

            # updating cluster centers
            cluster_weights = np.zeros(len(centers))
            cluster, cnts = np.unique(new_labels, return_counts=True)
            for k, c in enumerate(cluster):
                cluster_weights[c] = (1 / cnts[k])

            for k, l in enumerate(latents):
                c = self.ds.labels[k]
                centers[c] = centers[c] - cluster_weights[c] * (centers[c] - l)

        return centers

    def get_all_latents(self, dataloader):
        latents = list()
        for data in dataloader:
            inp = data[0].to(self.device)
            latent = self.model.get_latent(inp)
            latents.extend(latent.detach().cpu().numpy())
        return np.array(latents)

    def kmeans_friendly_loss_function(self, recon, x, latent_repr, cluster_center, reg):
        loss_recon = self.criterion(recon, x)
        loss_cluster = torch.sqrt(torch.sum((latent_repr - cluster_center) ** 2))

        return loss_recon + (reg * loss_cluster)
