from __future__ import annotations

import copy
import math
import time
import warnings

import numpy as np
import torch
from torch import nn

CATALOG = [
    ("persistence", "Persistence", "Reference baselines", "Last available observed concentration", "temporal"),
    ("seasonal_naive", "Seasonal naive", "Reference baselines", "Observation from the previous 24-hour cycle", "temporal"),
    ("arima", "ARIMA", "Reference baselines", "ARIMA(2,1,1) with training-only fitted parameters", "temporal"),
    ("random_forest", "Random forest", "Reference baselines", "80 regression trees over historical lag features", "temporal"),
    ("gradient_boosting", "Gradient boosting", "Reference baselines", "Histogram gradient boosting over historical lag features", "temporal"),
    ("lstm", "LSTM", "Temporal models", "Historical sequence → LSTM → forecast", "temporal"),
    ("cnn", "1D CNN", "Temporal models", "Historical sequence → temporal convolution → forecast", "temporal"),
    ("gru", "GRU", "Temporal models", "Historical sequence → GRU → forecast", "temporal"),
    ("bilstm", "BiLSTM", "Temporal models", "Bidirectional encoding of observed history → forecast", "temporal"),
    ("transformer", "Transformer", "Temporal models", "Positional encoding → historical self-attention → forecast", "temporal"),
    ("tft", "Temporal Fusion Transformer", "Temporal models", "Variable selection + static station embeddings + LSTM + gated attention; compact TFT-inspired implementation", "temporal"),
    ("cnn_lstm", "CNN-LSTM / LSTM + CNN", "Combined models", "Temporal CNN → LSTM → forecast", "temporal"),
    ("cnn_bilstm", "CNN + BiLSTM", "Combined models", "Temporal CNN → bidirectional historical encoder → forecast", "temporal"),
    ("lstm_attention", "LSTM + Attention", "Combined models", "LSTM states → learned temporal attention → forecast", "temporal"),
    ("transformer_lstm", "Transformer + LSTM", "Combined models", "Historical attention representations → LSTM → forecast", "temporal"),
    ("autoencoder_lstm", "Autoencoder-LSTM", "Combined models", "Training-only feature encoder + reconstruction → LSTM → forecast", "temporal"),
    ("stgnn", "Spatio-Temporal GNN", "Spatial & graph", "Fold-trained correlation graph → message passing → causal temporal CNN", "graph"),
    ("gcn_lstm", "GCN + LSTM", "Combined models", "Fold-trained graph convolution → LSTM → station forecasts", "graph"),
    ("convlstm", "ConvLSTM", "Spatial & graph", "Observed rectangular grid sequences → convolutional recurrent forecast", "grid"),
    ("unet", "U-Net", "Spatial & graph", "Past grid channels → encoder/decoder with skip connection → future field", "grid"),
    ("convlstm_attention", "ConvLSTM + Attention", "Combined models", "Convolutional recurrent grid states → temporal attention → forecast", "grid"),
]

def catalog(dataset=None):
    result=[]
    for identifier,name,group,architecture,modality in CATALOG:
        reason=None
        if dataset:
            if modality=="graph" and not dataset.get("graph_ready"):
                reason="Requires at least two aligned station histories."
            if modality=="grid" and not dataset.get("grid_shape"):
                reason="Requires observed stations on a complete regular rectangular grid; coordinates and target coverage must be supplied."
            if identifier=="seasonal_naive" and dataset.get("frequency")!="h":
                reason="The initial seasonal baseline requires hourly observations."
        result.append({"id":identifier,"name":name,"group":group,"architecture":architecture,"modality":modality,"compatible":reason is None,"reason":reason,"version":"1.0","implementation":"Reference method" if group=="Reference baselines" else "Inspired implementation; not a paper reproduction", "aliases":["LSTM + CNN"] if identifier=="cnn_lstm" else [], "runtime":"Fast" if group=="Reference baselines" else "CPU / optional GPU"})
    return result

class GatedResidual(nn.Module):
    def __init__(self, width):
        super().__init__();self.net=nn.Sequential(nn.Linear(width,width),nn.ELU(),nn.Linear(width,width));self.gate=nn.Linear(width,width);self.norm=nn.LayerNorm(width)
    def forward(self,x):return self.norm(x+torch.sigmoid(self.gate(x))*self.net(x))

class TemporalNet(nn.Module):
    def __init__(self, kind, features, hidden, nodes, lookback):
        super().__init__();self.kind=kind;self.nodes=nodes;self.aux=None
        self.project=nn.Linear(features,hidden)
        self.conv=nn.Sequential(nn.Conv1d(features,hidden,3,padding=2),nn.GELU(),nn.Conv1d(hidden,hidden,3,padding=2),nn.GELU())
        self.auto=nn.Sequential(nn.Linear(features,hidden),nn.GELU());self.decode=nn.Linear(hidden,features)
        self.variable=nn.Linear(features,features);self.static=nn.Embedding(nodes,hidden);self.gated=GatedResidual(hidden)
        self.attention=nn.Linear(hidden*(2 if kind in {"bilstm","cnn_bilstm"} else 1),1)
        enc=nn.TransformerEncoderLayer(hidden,4,hidden*2,dropout=0,batch_first=True)
        self.transformer=nn.TransformerEncoder(enc,1,enable_nested_tensor=False)
        in_width=hidden if kind in {"cnn_lstm","cnn_bilstm","transformer_lstm","autoencoder_lstm","tft"} else features
        recurrent=nn.GRU if kind=="gru" else nn.LSTM
        self.rnn=recurrent(in_width,hidden,batch_first=True,bidirectional=kind in {"bilstm","cnn_bilstm"})
        self.output=nn.Linear(hidden*(2 if kind in {"bilstm","cnn_bilstm"} else 1),1)
        positions=torch.arange(lookback,dtype=torch.float32)[:,None]
        divisor=torch.exp(torch.arange(0,hidden,2,dtype=torch.float32)*(-math.log(10000)/hidden))
        pos=torch.zeros(lookback,hidden);pos[:,0::2]=torch.sin(positions*divisor);pos[:,1::2]=torch.cos(positions*divisor)
        self.register_buffer('positions',pos)
    def forward(self,x):
        batch,length,nodes,features=x.shape
        seq=x.permute(0,2,1,3).reshape(batch*nodes,length,features);self.aux=None
        kind=self.kind
        if kind in {"cnn","cnn_lstm","cnn_bilstm"}:
            seq=self.conv(seq.transpose(1,2))[...,:length].transpose(1,2)
            if kind=="cnn":return self.output(seq.mean(1)).reshape(batch,nodes)
        if kind in {"transformer","transformer_lstm"}:
            seq=self.transformer(self.project(seq)+self.positions[:length])
            if kind=="transformer":return self.output(seq[:,-1]).reshape(batch,nodes)
        if kind=="autoencoder_lstm":
            original=seq;seq=self.auto(seq);self.aux=((self.decode(seq)-original)**2).mean()
        if kind=="tft":
            weights=torch.softmax(self.variable(seq),-1)
            station=self.static(torch.arange(nodes,device=x.device)).repeat(batch,1)[:,None,:]
            seq=self.gated(self.project(seq*weights)+station)
        states,_=self.rnn(seq)
        if kind in {"lstm_attention","tft"}:
            states=(torch.softmax(self.attention(states),dim=1)*states).sum(1)
            if kind=="tft":states=self.gated(states)
        else:states=states[:,-1]
        return self.output(states).reshape(batch,nodes)

class GraphNet(nn.Module):
    def __init__(self,kind,features,hidden,adjacency):
        super().__init__();self.kind=kind;self.register_buffer('adjacency',torch.tensor(adjacency,dtype=torch.float32))
        self.project=nn.Linear(features,hidden)
        self.rnn=nn.LSTM(hidden,hidden,batch_first=True)
        self.temporal=nn.Conv1d(hidden,hidden,3,padding=2)
        self.output=nn.Linear(hidden,1)
    def forward(self,x):
        spatial=torch.einsum('ij,btjf->btif',self.adjacency,x)
        h=torch.relu(self.project(spatial));b,t,n,d=h.shape
        seq=h.permute(0,2,1,3).reshape(b*n,t,d)
        if self.kind=='gcn_lstm':seq=self.rnn(seq)[0]
        else:seq=torch.relu(self.temporal(seq.transpose(1,2))[...,:t].transpose(1,2))
        return self.output(seq[:,-1]).reshape(b,n)

class ConvLSTMCell(nn.Module):
    def __init__(self,features,hidden):
        super().__init__();self.hidden=hidden;self.gates=nn.Conv2d(features+hidden,4*hidden,3,padding=1)
    def forward(self,x,h,c):
        i,f,o,g=self.gates(torch.cat([x,h],1)).chunk(4,1)
        c=torch.sigmoid(f)*c+torch.sigmoid(i)*torch.tanh(g)
        return torch.sigmoid(o)*torch.tanh(c),c

class GridNet(nn.Module):
    def __init__(self,kind,features,hidden,shape,lookback):
        super().__init__();self.kind=kind;self.shape=shape;self.hidden=hidden
        self.cell=ConvLSTMCell(features,hidden);self.att=nn.Linear(hidden,1)
        self.down=nn.Sequential(nn.Conv2d(features*lookback,hidden,3,padding=1),nn.GELU(),nn.Conv2d(hidden,hidden,3,padding=1),nn.GELU())
        self.bottom=nn.Sequential(nn.Conv2d(hidden,hidden*2,3,padding=1),nn.GELU())
        self.up=nn.Sequential(nn.Conv2d(hidden*3,hidden,3,padding=1),nn.GELU())
        self.output=nn.Conv2d(hidden,1,1)
    def forward(self,x):
        b,t,n,f=x.shape;rows,cols=self.shape
        grid=x.permute(0,1,3,2).reshape(b,t,f,rows,cols)
        if self.kind=='unet':
            skip=self.down(grid.reshape(b,t*f,rows,cols))
            bottom=self.bottom(nn.functional.avg_pool2d(skip,2))
            bottom=nn.functional.interpolate(bottom,size=(rows,cols),mode='bilinear',align_corners=False)
            state=self.up(torch.cat([skip,bottom],1))
        else:
            h=torch.zeros(b,self.hidden,rows,cols,device=x.device);c=torch.zeros_like(h);states=[]
            for step in grid.unbind(1):
                h,c=self.cell(step,h,c);states.append(h)
            state=h
            if self.kind=='convlstm_attention':
                sequence=torch.stack(states,1)
                weights=torch.softmax(self.att(sequence.mean((-1,-2))),dim=1)[...,None,None]
                state=(weights*sequence).sum(1)
        return self.output(state).reshape(b,n)

def make_network(kind,features,hidden,nodes,lookback,shape=None,adjacency=None):
    hidden=((hidden+3)//4)*4
    if kind in {'stgnn','gcn_lstm'}:return GraphNet(kind,features,hidden,adjacency)
    if kind in {'convlstm','convlstm_attention','unet'}:return GridNet(kind,features,hidden,shape,lookback)
    return TemporalNet(kind,features,hidden,nodes,lookback)

def graph_from_training(x):
    signal=np.mean(x,axis=(1,3))
    with np.errstate(divide='ignore',invalid='ignore'):
        adjacency=np.nan_to_num(np.corrcoef(signal.T),nan=0.0)
    adjacency=np.maximum(adjacency,0);np.fill_diagonal(adjacency,1)
    return adjacency/np.maximum(adjacency.sum(1,keepdims=True),1e-8)

def train_neural(network,x,y,vx,vy,config,checkpoint,control,progress):
    device='cuda' if config.device in {'cuda','auto'} and torch.cuda.is_available() else 'cpu'
    if config.device=='cuda' and device!='cuda':raise ValueError('CUDA is unavailable in this PyTorch installation. Select CPU or install the supported CUDA build.')
    torch.set_num_threads(2);network=network.to(device)
    optimizer=torch.optim.Adam(network.parameters(),lr=config.learning_rate)
    generator=torch.Generator().manual_seed(config.seed)
    start=0;best=float('inf');wait=0;best_state=copy.deepcopy(network.state_dict());losses=[]
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location=device,weights_only=False)
        network.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        start=saved['epoch'];best=saved['best'];wait=saved['wait'];best_state=saved['best_state'];losses=saved['losses'];generator.set_state(saved['generator'])
        torch.set_rng_state(saved['torch_rng'])
    X=torch.as_tensor(x,dtype=torch.float32);Y=torch.as_tensor(y,dtype=torch.float32)
    VX=torch.as_tensor(vx,dtype=torch.float32);VY=torch.as_tensor(vy,dtype=torch.float32)
    def validation_loss():
        network.eval();total=0.;count=0
        with torch.no_grad():
            for offset in range(0,len(VX),config.batch_size):
                target=VY[offset:offset+config.batch_size].to(device);pred=network(VX[offset:offset+config.batch_size].to(device));mask=torch.isfinite(target)
                total+=float(((pred[mask]-target[mask])**2).sum());count+=int(mask.sum())
        if not count:raise ValueError('Validation partition has no observed targets.')
        return total/count
    for epoch in range(start,config.epochs):
        control();network.train();total=0.;count=0
        order=torch.randperm(len(X),generator=generator)
        for indices in order.split(config.batch_size):
            xb=X[indices].to(device);yb=Y[indices].to(device);mask=torch.isfinite(yb)
            if not mask.any():continue
            optimizer.zero_grad();pred=network(xb);loss=((pred[mask]-yb[mask])**2).mean()
            aux=getattr(network,'aux',None)
            if aux is not None:loss=loss+.1*aux
            loss.backward();nn.utils.clip_grad_norm_(network.parameters(),1);optimizer.step()
            total+=float(loss.detach())*int(mask.sum());count+=int(mask.sum())
        val=validation_loss();train=total/max(count,1);losses.append({'epoch':epoch+1,'train_loss':train,'validation_loss':val})
        if val<best:best=val;best_state=copy.deepcopy(network.state_dict());wait=0
        else:wait+=1
        torch.save({'model':network.state_dict(),'optimizer':optimizer.state_dict(),'epoch':epoch+1,'best':best,'wait':wait,'best_state':best_state,'losses':losses,'generator':generator.get_state(),'torch_rng':torch.get_rng_state()},checkpoint)
        progress(epoch+1,train,val)
        control()
        if wait>=config.patience:break
    network.load_state_dict(best_state);network.eval()
    return network,losses,device

def predict_neural(net,x,device,batch_size):
    predictions=[]
    with torch.no_grad():
        for offset in range(0,len(x),batch_size):
            pred=net(torch.as_tensor(x[offset:offset+batch_size],dtype=torch.float32,device=device))
            predictions.append(pred.cpu().numpy())
    return np.concatenate(predictions)

def fit_classical(kind,x,y,seed):
    from sklearn.ensemble import RandomForestRegressor,HistGradientBoostingRegressor
    estimator=RandomForestRegressor(n_estimators=80,min_samples_leaf=2,n_jobs=2,random_state=seed) if kind=='random_forest' else HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,random_state=seed)
    flat=x.transpose(0,2,1,3).reshape(-1,x.shape[1]*x.shape[3]);target=y.reshape(-1);mask=np.isfinite(target)
    if mask.sum()<24:raise ValueError('Training needs at least 24 observed targets.')
    estimator.fit(flat[mask],target[mask]);return estimator

def predict_classical(estimator,x):
    flat=x.transpose(0,2,1,3).reshape(-1,x.shape[1]*x.shape[3])
    return estimator.predict(flat).reshape(len(x),x.shape[2])
