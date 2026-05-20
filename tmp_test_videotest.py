import sys
sys.path.insert(0, '.')
from lstm_veri_topla import videodan_kare_dizisi_cek
arr = videodan_kare_dizisi_cek('videolar/abartmak.mp4')
print('frames', arr.shape)
