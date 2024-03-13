# Dataset_Custom formatting


import numpy as np
import pandas as pd
from datetime import datetime
import os
import seaborn as sns
from matplotlib.patches import Patch
import matplotlib.pyplot as plt
from matplotlib import ticker
sns.set_context('paper')
sns.set_style('ticks')

#%% load metadata
FFP_model = ['KM','Hsieh','Kljun']
model_lst = ['KM','Hsieh','Kljun','R500m','R250m']
type_lst = ['DBF','ENF','DNF','nonF','W']
type_col_lst = [t+'_FFP_frac' for t in type_lst]
VIs_lst = ['NDVI','NIRv','kNDVI']
tower_lst = ['PF' + s for s in ['b', 'e', 'g', 'h', 'i', 'j', 'k', 'L', 'm', 'n','p', 'q', 't']]# 'd','c','s'
FFP_model = ['KM', 'Hsieh', 'Kljun']
model_lst = ['KM', 'Hsieh', 'Kljun', 'R500m', 'R250m']
type_lst = ['DBF', 'ENF', 'DNF', 'nonF', 'W']
type_col_lst = [t + '_FFP_frac' for t in type_lst]
VIs_lst = ['NDVI', 'NIRv', 'kNDVI']
tower_lst = ['PF' + s for s in ['b', 'e', 'g', 'h', 'i', 'j', 'k', 'L', 'm', 'n', 'p', 'q', 't']]  # 'd','c','s'
type_lst = ['DBF', 'ENF', 'DNF', 'nonF', 'W']
sat = 'PS'
model = model_lst[0]
VI = VIs_lst[0]
PFT_FFP_frac_lst = ['frac_{}_FFP'.format(t) for t in type_lst]
PFT_R500m_frac_lst = ['frac_{}_R500m'.format(t) for t in type_lst]
PFT_VIs_lst = ['{}_{}_{}_FFP'.format(t, VI,sat) for t in type_lst for VI in VIs_lst]
VIs_FFP_lst = ['{}_{}_FFP'.format(VI,sat) for VI in VIs_lst]
VIs_R500m_lst = ['{}_{}_R500m'.format( VI,sat) for VI in VIs_lst]
Flux_features = ['Tsoil', 'Rg', 'Tair', 'rH', 'VPD']

FEATURE_DICT = {
    'Tradition': Flux_features + VIs_R500m_lst,
    'Dynamic': Flux_features + VIs_FFP_lst,
    'Tradition+FracFixed': Flux_features + VIs_R500m_lst + PFT_R500m_frac_lst,
    'Dynamic+FracDyna': Flux_features + VIs_FFP_lst +PFT_VIs_lst + PFT_FFP_frac_lst,
}

df_RF_ffp = pd.read_csv('D:\\Ziyu\\CHEESEHEAD_PFT_summary.csv')
df_RF_ffp = df_RF_ffp[df_RF_ffp.FFPmodel==model]
df_FLUX = pd.read_csv('D:\\Ziyu\\CHEESEHEAD_FLUX.csv')
data_ = df_RF_ffp.merge(df_FLUX, left_on=['HalfHour', 'DOY', 'site'], right_on=['HalfHour', 'DOY', 'site'], how='inner')
data_.NEE.describe()
# %%
# Ustar threshold columns: Ustar, Ustar_uStar_Thres, Ustar_U75_Tres,Ustar_U25_Tres,Ustar_U95_Tres,Ustar_U05_Tres, FP_QC(0:bad,1:good)
# time columns: year{YYYY},DOY{DDD},HalfHour{HH.H} season{20190MM}, TIMESTAMP_START{YYYMMDDHHHH},TIMESTAMP_END{YYYMMDDHHHH}=HalfHour=DateTime,
# filter outliners
for c in ['NEE','Tair','VPD','Rg']:
    outliner_res = data_.groupby(['DOY'])[c].quantile([0.05, 0.95]).unstack(level=1)
    data_filter = data_.loc[((outliner_res.loc[data_['DOY'], 0.05] < data_[c].values)
            & (data_[c].values < outliner_res.loc[data_['DOY'], 0.95])).values]
# data_night = data_[((data_.HalfHour>=18)|(data_.HalfHour<=6))&(data_.Rg<200)]
# data_noon = data_[((data_.HalfHour>=10)&(data_.HalfHour<=14))&(data_.Rg>200)]
# %% non-stantionary NEE site level
# TIMESTAMP: {MMDDHHHH}
# data_filter =df_FLUX
data_filter['TIMESTAMP_norm'] = data_filter.DOY + data_filter.HalfHour/24


# %%

win = 37 # must be odd number
for c in ['NEE','Tair','VPD','Rg']:
# for c in ['Reco_DT_uStar','Reco_uStar','GPP_DT_uStar','GPP_uStar_f','FP_NEEnight_U50']:
# for c in ['FP_VARnight','FP_dRecPar']:
# for c in ['Tair','NEW_FP_Temp','VPD','NEW_FP_VPD']:
#     data_filter['TIMESTAMP_norm'][data_filter[c]!=np.nan]==np.nan
    smoothed = data_filter[data_filter.DOY>200].sort_values('TIMESTAMP_norm').groupby(['site'])[[c,'TIMESTAMP_norm']].apply(lambda d: [d.TIMESTAMP_norm,d[c],np.convolve(d[c], np.ones(win)/win)[int(win/2):-int(win/2)]])

    for S in tower_lst[1:2 ]:
        plt.scatter(smoothed[S][0],smoothed[S][1],alpha=0.1)
        mask = smoothed[S][2][:]!=np.nan
        plt.plot(smoothed[S][0][mask],smoothed[S][2][mask],lw=2,label='{} (n={})'.format(S,sum(smoothed[S][1]!=np.nan)))
    plt.legend(frameon=False)
    plt.xlabel('DOY')
    plt.ylabel(c)
    plt.title('soomthed (window = {0:.1f}hrs = {1:.1f}days)'.format(win/2,win/48))
    plt.show()


# %% Dataset_Custom:  df_raw.columns: ['date', ...(other features), target feature]
data_filter['date'] = data_filter.DateTime
# data_filter = data_filter.sample(frac=1)
# data_filter['train'] = [1 if i > len(data_filter)/3 else 0 for i in range(len(data_filter))]

feature_list = FEATURE_DICT
# data_filter  = data_filter
data_filter[data_filter.DOY>200][FEATURE_DICT['Dynamic+FracDyna']+['NEE','date']].to_csv('./dataset/NEE_v1.csv')
# Dataset_Custom('./dataset', flag='train', size=None,
#                  features='S', data_path='CNEE_v1.csv',
#                  target='NEE', scale=True, timeenc=0, freq='h')
# %%
import pandas as pd
# data = pd.read_csv('./dataset/NEE_v1.csv')
# Fs = FEATURE_DICT['Tradition']
Fs = FEATURE_DICT['Dynamic+FracDyna']

data = pd.DataFrame(data_filter[Fs+['NEE']])
data.fillna(data.mean(),inplace=True)
data['date'] = data_filter.DateTime
data_site = data[data_filter.site=='PFb'][Fs+['NEE','date']]
data_site.to_csv('./dataset/NEE_fillna_PFb_Dynamic.csv')
# %%
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True' # state before import torch
import torch

dataset_name = 'NEE_fillna_PFb_Tradition.csv'
n_col = pd.read_csv('./dataset/'+dataset_name).columns.size-1
def run_model():
    os.system(
        "python -u run.py \
      --is_training 1 \
      --data custom \
      --root_path ./dataset/ \
      --data_path {} \
      --target NEE\
      --freq h\
      --model ns_Transformer \
      --features M \
      --train_epochs 10\
      --learning_rate 0.01 \
      --model_id NEE_8_120_tra \
      --seq_len 8 \
      --pred_len 120 \
      --label_len 4 \
      --moving_avg 4 \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in {} \
      --dec_in {} \
      --c_out {} \
      --des Exp_h64_l2 \
      --p_hidden_dims 64 64 \
      --p_hidden_layers 1 \
      --itr 1\
      --num_worker 0\
      --seed 2024".format(dataset_name,n_col,n_col,n_col))


if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    run_model()

# python -u run.py --is_training 1 --root_path ./dataset/ --data custom --data_path NEE_v1.csv --model_id NEE_16_240 --model ns_Transformer --features M --target NEE --seq_len 16 --label_len 48 --pred_len 240 --e_layers 2 --d_layers 1 --factor 3 --enc_in 21 --dec_in 21 --c_out 21 --gpu 1 --des 'Exp_h128_l2' --p_hidden_dims 128 128 --p_hidden_layers 2 --itr 1 & --seed 2024

