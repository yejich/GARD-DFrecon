"""Wait for a selected physical GPU to have room, then resume qualitative eval."""
import argparse,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[3]
print(f'Waiting for physical GPU {a.gpu}: at least 24000 MiB free',flush=True)
while True:
 result=subprocess.check_output(['nvidia-smi','-i',a.gpu,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
 if int(result.strip())>=24000:break
 time.sleep(15)
env=os.environ.copy();env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=a.gpu,HF_HUB_CACHE='/mnt/nvme1n1/hyuna/hf_cache/hub',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',MPLCONFIGDIR='/tmp/gard_registry_qual_mpl')
print(f'Starting qualitative eval on GPU {a.gpu}',flush=True)
os.environ.update(env)
os.execv(str(root/'.venv/bin/python'),[str(root/'.venv/bin/python'),'-u',str(root/'scripts/eval/registry_qual/run.py'),'--config',a.config,'--checkpoint',a.checkpoint,'--output',a.output])
