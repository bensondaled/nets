'''
goal: using trial_timing from puffs data, cache each trial's binned L and R values for faster lookup
'''

##

bin_width = .200

##

data_file = '/Users/ben/data/puffs/merged/20180411_pillow/data_20180411.h5'
with pd.HDFStore(data_file) as h:
    t = h.trials
    tt = h.trials_timing

def add_uid(t):
    
    seshs = t.session.values
    if 'idx' in t.columns:
        trials = t.idx.values
    elif 'trial' in t.columns:
        trials = t.trial.values

    uids = []

    for idx,(sh,tr) in enumerate(zip(seshs,trials)):
        uid = '{}-{}'.format(sh,int(tr))
        uids.append(uid)

    t.loc[:,'uid'] = uids

    return t

t = add_uid(t)
tt = add_uid(tt)

ttgb = tt.groupby('uid')

##
        
bins = np.arange(0, 3.8+1e-10, bin_width)

durs = []
densities = []
choices = []
uids = []

for idx,(subj,sesh,tidx,ch,dur,uid) in enumerate(t[['subj','session','idx','choice','dur','uid']].values): 

    tti = ttgb.get_group(uid)
     
    # compute binned stim counts
    density = np.zeros([2,len(bins)-1], dtype=int)
    binl,_ = np.histogram(tti[tti.side==0].time, bins=bins)
    binr,_ = np.histogram(tti[tti.side==1].time, bins=bins)

    density[0] = binl
    density[1] = binr

    durs.append(dur)
    densities.append(density)
    choices.append(ch)
    uids.append(uid)
    
    if idx%250==0:    print(idx,'/',len(t))

##

np.savez_compressed('trials', dur=durs, timing=densities, choice=choices, uid=uids)

##
