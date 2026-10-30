"""Sequential single-GPU suite: finish Hypersim before real samples."""
import os,sys,json,subprocess,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=ROOT/'result_eval/20260923_quant_eval'
stages=[('hypersim_score_gpu5',['score.py','--dataset','hypersim','--groups','ai_001_007_group00','ai_001_008_group00','ai_001_009_group00','ai_001_010_group00']),('real_da3_gpu5',['infer.py','--dataset','real','--backend','da3']),('real_vgtw_gpu5',['infer.py','--dataset','real','--backend','vgtw']),('real_score_gpu5',['score.py','--dataset','real'])]
status={'gpu':os.environ.get('CUDA_VISIBLE_DEVICES'),'status':'running','completed':[],'started':datetime.datetime.now().isoformat()}
for name,args in stages:
 status['stage']=name;(OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2))
 print('STAGE',name,flush=True)
 with (OUT/f'{name}.log').open('a') as log:
  rc=subprocess.call([sys.executable,'-u',str(ROOT/'scripts/eval/quant_eval'/args[0]),*args[1:]],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 if rc:
  status.update(status='failed',returncode=rc);(OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2));sys.exit(rc)
 status['completed'].append(name)
status.update(status='complete',finished=datetime.datetime.now().isoformat());(OUT/'pipeline_status.json').write_text(json.dumps(status,indent=2));print('ALL STAGES COMPLETE',flush=True)
