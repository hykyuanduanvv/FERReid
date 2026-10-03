"""Repeat identity-grouped domain probes on saved trace representations (CPU only)."""
import argparse, json
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from threadpoolctl import threadpool_limits

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('directory')
    a=parser.parse_args(); root=Path(a.directory)
    z=np.load(root/'pooled_representations.npz')
    y=z['labels']; groups=z['buckets']; nclass=len(np.unique(y)); result={}
    for key in z.files:
        if key in ['labels','buckets']: continue
        x=z[key]; scores=[]; null=[]; predictions=[]
        for repeat in range(3):
            # Different held-out identity buckets in each domain, each bucket tested once per repeat.
            rng=np.random.RandomState(1800+repeat)
            maps=[rng.permutation(3) for _ in range(nclass)]
            folds=np.array([maps[c][g] for c,g in zip(y,groups)])
            pred=np.full(len(y),-1)
            for fold in range(3):
                tr=folds!=fold; te=~tr
                pre=make_pipeline(StandardScaler(),PCA(n_components=min(16,tr.sum()-1,x.shape[1]),svd_solver='full'))
                xt=pre.fit_transform(x[tr]); xv=pre.transform(x[te])
                clf=LogisticRegression(C=1.,max_iter=2000).fit(xt,y[tr])
                p=clf.predict(xv); pred[te]=p
                scores.append(float(balanced_accuracy_score(y[te],p)))
                for perm in range(5):
                    shuffled=np.random.RandomState(9000+repeat*100+fold*10+perm).permutation(y[tr])
                    cp=LogisticRegression(C=1.,max_iter=2000).fit(xt,shuffled).predict(xv)
                    null.append(float(balanced_accuracy_score(y[te],cp)))
            predictions.append(pred.tolist())
        result[key]=dict(mean_accuracy=float(np.mean(scores)),fold_accuracies=scores,
            repeat_accuracy=[float(balanced_accuracy_score(y,p)) for p in predictions],
            shuffled_label_mean=float(np.mean(null)),shuffled_label_sd=float(np.std(null)),
            shuffled_label_max=float(np.max(null)),
            confusion_matrix=sum(confusion_matrix(y,p,labels=range(nclass)) for p in predictions).tolist(),
            note='3 repeats x 3 grouped folds; contexts overlap within bucket only; repeats are correlated')
        print(key,round(result[key]['mean_accuracy'],4),round(result[key]['shuffled_label_mean'],4),flush=True)
    (root/'probes_repeated.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    geom=json.loads((root/'geometry.json').read_text())
    layers=sorted(int(k.split('_')[1]) for k in geom if k.startswith('query_') and k.split('_')[1].isdigit())
    fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
    for prefix,label in [('query','Query slots'),('context','Context tokens (pooled)')]:
        axes[0].plot(layers,[geom[f'{prefix}_{i:02}']['rho'] for i in layers],label=label)
        axes[1].plot(layers,[result[f'{prefix}_{i:02}']['mean_accuracy'] for i in layers],label=label)
    axes[0].set(xlabel='Hidden-state index (0 = input)',ylabel='Variation / total energy',title='Representation geometry')
    axes[0].legend(); axes[1].axhline(1/nclass,color='grey',linestyle='--',label='Chance')
    axes[1].set(xlabel='Hidden-state index',ylabel='Grouped probe balanced accuracy',ylim=(0,1.03),title='Known-domain classification'); axes[1].legend()
    keys=['query_%02d'%max(layers),'prompt','query_last_masked','prompt_masked','fixed_image_features']
    labels=['Last query','Prompt','Masked query','Masked prompt','Image features']
    axes[2].bar(labels,[geom[k]['rho'] for k in keys])
    axes[2].tick_params(axis='x',rotation=35)
    axes[2].set(ylabel='Variation / total energy',title='Geometry is not retrieval utility')
    fig.savefig(root/'trace_overview.png',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    with threadpool_limits(limits=4): main()
